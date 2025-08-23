#!/usr/bin/env python3
"""
Pedalboard + CLAP smart exploration

"""
import argparse, os, json, csv
import re

import librosa
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
import multiprocessing as mp
import time
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x
from dataclasses import dataclass
from typing import List, Dict, Tuple

from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

import numpy as np
import soundfile as sf

from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowpassFilter,
    LowShelfFilter,
    HighShelfFilter,
    PeakFilter,
    Reverb,
    Delay,
    Distortion,
    PitchShift,
    Bitcrush,
)

import torch
from transformers import ClapProcessor, ClapModel

# -------------------------------
# Utility
# -------------------------------
def load_audio_mono(path: str, target_sr: int = 48000):
    # 1. librosa를 사용해 오디오를 로드합니다. sr은 파일의 원본 샘플레이트입니다.
    # mono=True 옵션으로 바로 모노 채널로 변환합니다.
    audio, sr = librosa.load(path, sr=None, mono=True)
    
    # 2. 샘플레이트가 다를 경우, librosa의 리샘플링 기능을 사용합니다.
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    # 3. pedalboard가 요구하는 형태로 차원을 맞춰줍니다. (1, num_samples)
    audio = audio[np.newaxis, :]
    
    return audio.astype(np.float32), sr

def save_audio(path: str, audio: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio.T, sr, subtype="PCM_16")
    
class EarlyStopper:
    """
    Custom early stopping callback.
    Stops the optimization if the best score hasn't improved by at least `delta`
    in the last `n_best` iterations.
    """
    def __init__(self, delta=0.0, n_best=10):
        self.delta = delta
        self.n_best = n_best
        self.best_func = np.inf
        self.counter = 0

    def __call__(self, res):
        # res.fun is the best score found so far
        if res.fun < self.best_func - self.delta:
            self.best_func = res.fun
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.n_best:
            print(f"\nStopping early because score hasn't improved in {self.n_best} iterations.")
            # Returning True stops the optimization
            return True
        
def safe_folder_name(text):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', text.strip().replace(' ', '_'))

# -------------------------------
# CLAP scoring
# -------------------------------
def get_clap_model(model_name: str, device: str):
    model = ClapModel.from_pretrained(model_name).to(device)
    processor = ClapProcessor.from_pretrained(model_name)
    return model, processor

def clap_score_batch(audio_list: list[np.ndarray], sr: int, prompt: str, model, processor, device: str) -> list[float]:
    wavs = [a.squeeze().astype(np.float32) for a in audio_list]
    # NOTE: The processor is smart enough to handle a single prompt for a batch of audio.
    # We pass the single prompt string directly.
    inputs = processor(text=prompt, audios=wavs, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # 1. 텍스트와 오디오 임베딩을 각각 추출
        text_features = model.get_text_features(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        audio_features = model.get_audio_features(input_features=inputs['input_features'])

        # 2. L2 정규화
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=-1)

        scores_tensor = audio_features @ text_features.T
        
        scores = scores_tensor.squeeze().cpu().numpy().tolist()
        
    return [scores] if not isinstance(scores, list) else scores

def clap_score_batch_negative(audio_list: list[np.ndarray], sr: int, prompts: str or list[str], model, processor, device: str) -> list[float]: # type: ignore
    """
    Calculates CLAP scores. Handles both a single prompt for all audio clips
    and a list of prompts corresponding to each audio clip.
    """
    wavs = [a.squeeze().astype(np.float32) for a in audio_list]
    
    # prompts가 단일 문자열일 경우, 오디오 개수만큼 복제하여 리스트로 만듭니다.
    if isinstance(prompts, str):
        prompts = [prompts] * len(wavs)
        
    inputs = processor(text=prompts, audios=wavs, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_features = model.get_text_features(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        audio_features = model.get_audio_features(input_features=inputs['input_features'])

        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=-1)

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분이 핵심 수정사항 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 각 오디오와 그에 해당하는 텍스트의 유사도를 직접 계산 (element-wise dot product)
        # (audio_features * text_features)는 각 원소별 곱셈
        # .sum(dim=1)은 각 임베딩 벡터의 내적을 계산
        scores_tensor = (audio_features * text_features).sum(dim=1)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 이 부분이 핵심 수정사항 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        scores = scores_tensor.cpu().numpy().tolist()
        
    return [scores] if not isinstance(scores, list) else scores

def clap_score(audio_mono: np.ndarray, sr: int, prompt: str, model, processor, device: str) -> float:
    return clap_score_batch([audio_mono], sr, prompt, model, processor, device)[0]
# -------------------------------
# Plugin renderers
# -------------------------------
def build_eq_chain(mode: str, low_cut=80.0, high_cut=14000.0, q=1.0, gains: Dict[str, float] = None, peak1_freq=200.0, peak2_freq=1000.0, peak3_freq=5000.0):
    default_gains = {'low_shelf': 3.0, 'high_shelf': -2.0, 'peak1': -1.5, 'peak2': 2.0, 'peak3': 1.0}
    if gains is None:
        gains = default_gains.copy()
    else:
        # 누락된 키는 기본값으로 채움
        for k, v in default_gains.items():
            if k not in gains:
                gains[k] = v

    chain = []
    # 1. 'pass-shelf' 같은 문자열을 '-' 기준으로 분리하여 low_mode와 high_mode를 결정합니다.
    try:
        low_mode, high_mode = mode.split('-')
    except ValueError:
        # 혹시 모를 에러에 대비해 기본값 설정
        low_mode, high_mode = "pass", "pass"

    # 2. low_mode에 따라 저역 필터를 추가합니다.
    if low_mode == "pass":
        chain.append(HighpassFilter(cutoff_frequency_hz=low_cut))
    elif low_mode == "shelf":
        chain.append(LowShelfFilter(cutoff_frequency_hz=low_cut, gain_db=gains['low_shelf'], q=q))

    # 3. high_mode에 따라 고역 필터를 추가합니다.
    if high_mode == "pass":
        chain.append(LowpassFilter(cutoff_frequency_hz=high_cut))
    elif high_mode == "shelf":
        chain.append(HighShelfFilter(cutoff_frequency_hz=high_cut, gain_db=gains['high_shelf'], q=q))
    
    # 4. 3개의 피크 필터는 그대로 유지됩니다.
    chain.extend([
        PeakFilter(cutoff_frequency_hz=peak1_freq, gain_db=gains['peak1'], q=q),
        PeakFilter(cutoff_frequency_hz=peak2_freq, gain_db=gains['peak2'], q=q),
        PeakFilter(cutoff_frequency_hz=peak3_freq, gain_db=gains['peak3'], q=q)
    ])
    return chain

# variant builder given params
def render(audio, sr, config: Dict):
    order = [
            "EQ",          # 불필요한 주파수 제거
            "Distortion",  # 톤 캐릭터
            "Bitcrush",    # lo-fi 질감
            "PitchShift",  # 피치 변환
            "Delay",       # 시간적 공간감
            "Reverb"       # 최종 공간감
        ]
        # config를 order에 따라 정렬
    config_sorted = sorted(config, key=lambda x: order.index(x["type"]) if x["type"] in order else 99)
    board = []
    for fx in config_sorted:
        fx_type = fx["type"]
        if fx_type == "Distortion":
            board.append(Distortion(drive_db=fx["drive_db"]))
        elif fx_type == "EQ":
            board.extend(build_eq_chain(
            fx["mode"], fx["low_cut"], fx["high_cut"], fx.get("q", 1.0), fx.get("gains"),
            fx.get("peak1_freq", 200.0), fx.get("peak2_freq", 1000.0), fx.get("peak3_freq", 5000.0)
        ))
        elif fx_type == "Reverb":
            board.append(Reverb(room_size=fx["room_size"], damping=fx.get("damping", 0.4), wet_level=fx.get("wet_level", 0.12)))
        elif fx_type == "Delay":
            board.append(Delay(delay_seconds=fx["delay"], feedback=0.25))
        elif fx_type == "PitchShift":
            board.append(PitchShift(semitones=fx["semitones"]))
        elif fx_type == "Bitcrush":
            board.append(Bitcrush(bit_depth=fx["bit_depth"]))
    pb = Pedalboard(board)
    return pb(audio, sr)

# -------------------------------
# Binary search refinement
# -------------------------------
def binary_search_param(audio, sr, base_config, fx_name, param_name, lo, hi, model, processor, device, prompt, scale='linear', steps=7):
    best_val = lo
    initial_processed = render(audio, sr, base_config)
    best_score = clap_score(initial_processed, sr, prompt, model, processor, device)
    search_lo, search_hi = lo, hi
    if scale == 'log':
        search_lo, search_hi = np.log10(max(lo, 1e-6)), np.log10(max(hi, 1e-6))
    for _ in range(steps):
        mid_point = (search_lo + search_hi) / 2
        current_val = 10**mid_point if scale == 'log' else mid_point
        config = [dict(f, **{'gains': f['gains'].copy()}) if 'gains' in f else dict(f) for f in base_config]
        found = False
        for f in config:
            if f["type"] == fx_name:
                keys = param_name.split('.')
                if len(keys) == 2:
                    main_key, sub_key = keys
                    # Check if the 'gains' dictionary exists. If not, create it as an empty dictionary.
                    if main_key not in f:
                        f[main_key] = {}
                    f[main_key][sub_key] = current_val
                else: f[param_name] = current_val
                found = True
                break
        if not found: return best_val, best_score
        processed = render(audio, sr, config)
        score = clap_score(processed, sr, prompt, model, processor, device)
        if score > best_score:
            best_score, best_val = score, current_val
            if current_val > (10**search_lo if scale == 'log' else search_lo) + (search_hi - search_lo) * 0.5: search_lo = mid_point
            else: search_hi = mid_point
        else:
            if current_val > (10**search_lo if scale == 'log' else search_lo) + (search_hi - search_lo) * 0.5: search_hi = mid_point
            else: search_lo = mid_point
    return best_val, best_score

def batched_search_param(audio, sr, base_config, fx_name, param_name, lo, hi, model, processor, device, prompt, scale='linear', steps=3, batch_size=8):
    """
    Performs a batched search for the best parameter value.
    In each step, it evaluates `batch_size` candidates simultaneously.
    """
    best_val_overall = lo
    
    # Get the initial score to compare against
    initial_processed = render(audio, sr, base_config)
    best_score_overall = clap_score(initial_processed, sr, prompt, model, processor, device)

    current_lo, current_hi = lo, hi
    
    for _ in range(steps):
        # 1. 현재 탐색 범위 내에서 `batch_size`개의 후보 값을 생성합니다.
        if scale == 'log':
            test_vals = np.logspace(np.log10(max(current_lo, 1e-6)), np.log10(max(current_hi, 1e-6)), batch_size)
        else:
            test_vals = np.linspace(current_lo, current_hi, batch_size)
            
        # 2. 각 후보 값에 대한 오디오 배치를 생성합니다.
        audio_batch = []
        configs_batch = []
        for val in test_vals:
            config = [dict(f, **{'gains': f['gains'].copy()}) if 'gains' in f else dict(f) for f in base_config]
            for f in config:
                if f["type"] == fx_name:
                    keys = param_name.split('.')
                    if len(keys) == 2:
                        if keys[0] not in f: f[keys[0]] = {}
                        f[keys[0]][keys[1]] = val
                    else:
                        f[param_name] = val
                    break
            audio_batch.append(render(audio, sr, config))
        
        # 3. 오디오 배치를 "단 한 번에" 평가합니다.
        scores_batch = clap_score_batch(audio_batch, sr, prompt, model, processor, device)
        
        # 4. 이번 배치에서 가장 좋았던 값과 점수를 찾습니다.
        best_idx_in_batch = np.argmax(scores_batch)
        best_score_in_batch = scores_batch[best_idx_in_batch]
        best_val_in_batch = test_vals[best_idx_in_batch]
        
        # 5. 전체 최고 기록을 업데이트합니다.
        if best_score_in_batch > best_score_overall:
            best_score_overall = best_score_in_batch
            best_val_overall = best_val_in_batch
        
        # 6. 다음 스텝을 위해, 이번 배치에서 가장 좋았던 값 주변으로 탐색 범위를 좁힙니다.
        width = (current_hi - current_lo) * 0.4 # 범위를 40%로 줄여서 집중
        current_lo = max(lo, best_val_in_batch - width / 2)
        current_hi = min(hi, best_val_in_batch + width / 2)

    return best_val_overall, best_score_overall

# ==============================================================================
# Refinement Worker Function (for Parallel Processing)
# ==============================================================================
def refine_candidate(args_dict):
    """Encapsulated refinement logic for a single candidate to be run in a separate process."""
    # Unpack all arguments
    rank = args_dict['rank']
    initial_score = args_dict['initial_score']
    best_initial_config = args_dict['best_initial_config']
    audio = args_dict['audio']
    sr = args_dict['sr']
    PARAM_RANGES = args_dict['PARAM_RANGES']
    model_name = args_dict['model_name']
    prompt = args_dict['prompt']
    outdir = args_dict['outdir']

    # Each process must initialize its own model instance
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_clap_model(model_name, device)
    
    tuned_cfg = [p.copy() for p in best_initial_config]
    
    # 2-Phase Iterative Search
    for i in range(2):
        for fx_idx, initial_fx in enumerate(best_initial_config):
            fx_type = initial_fx["type"]
            if fx_type not in PARAM_RANGES: continue
            
            is_active = (
                (fx_type == "Distortion" and initial_fx.get('drive_db', 0) > 0) or
                (fx_type == "PitchShift" and initial_fx.get('semitones', 0) != 0) or
                (fx_type == "Bitcrush" and initial_fx.get('bit_depth', 16) < 16) or
                (fx_type == "Delay" and initial_fx.get('delay', 0) > 0) or
                (fx_type == "Reverb" and initial_fx.get('wet_level', 0) > 0) or
                (fx_type == "EQ")
            )
            if not is_active: continue

            for param_name, search_info in PARAM_RANGES[fx_type].items():
                tuned_fx = tuned_cfg[fx_idx]
                keys = param_name.split('.')
                if len(keys) == 2: current_val = tuned_fx.get(keys[0], {}).get(keys[1], search_info['lo'])
                else: current_val = tuned_fx.get(param_name, search_info['lo'])
                
                if i == 1:
                    width = (search_info['hi'] - search_info['lo']) * 0.3
                    lo, hi = max(search_info['lo'], current_val - width / 2), min(search_info['hi'], current_val + width / 2)
                else:
                    lo, hi = search_info['lo'], search_info['hi']

                val, _ = binary_search_param(
                    audio, sr, tuned_cfg, fx_type, param_name, lo, hi,
                    model, processor, device, prompt, search_info['scale']
                )

                if len(keys) == 2:
                    if keys[0] not in tuned_fx: tuned_fx[keys[0]] = {}
                    tuned_fx[keys[0]][keys[1]] = val
                else:
                    tuned_fx[param_name] = val

    # Calculate final score and save audio
    final_audio = render(audio, sr, tuned_cfg)
    final_score = clap_score(final_audio, sr, prompt, model, processor, device)
    save_audio(os.path.join(outdir, f"final_best_rank_{rank+1}.wav"), final_audio, sr)

    # Return a dictionary with all results
    return {
        "rank": rank + 1,
        "initial_score": initial_score,
        "final_score": final_score,
        "plugins": tuned_cfg
    }

def refine_candidate_bayesian(args_dict):
    """Refines a candidate using Bayesian Optimization."""
    initial_config = args_dict['initial_config']
    audio = args_dict['audio']
    sr = args_dict['sr']
    PARAM_RANGES = args_dict['PARAM_RANGES']
    model_name = args_dict['model_name']
    prompt = args_dict['prompt']
    outdir = args_dict['outdir']
    top_n = args_dict['top_n']
    n_calls = args_dict['n_calls']
    use_negative = args_dict['use_negative']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_clap_model(model_name, device)

    # 1. 탐색 공간(Search Space) 정의
    search_space = []
    param_names = []
    
    for fx in initial_config:
        fx_type = fx["type"]
        if fx_type not in PARAM_RANGES: continue
        
        # EQ가 아닐 경우에만 'activation' 파라미터를 추가
        if fx_type != "EQ":
            search_space.append(Real(0.0, 1.0, name=f"{fx_type}__activation"))
            param_names.append(f"{fx_type}__activation")

        for p_name, p_info in PARAM_RANGES[fx_type].items():
            full_param_name = f"{fx_type}__{p_name}"
            param_names.append(full_param_name)
            if p_info.get('type') == 'categorical':
                search_space.append(Categorical(p_info['choices'], name=full_param_name))
            elif p_info.get('scale') == 'log':
                search_space.append(Real(p_info['lo'], p_info['hi'], prior='log-uniform', name=full_param_name))
            else:
                search_space.append(Real(p_info['lo'], p_info['hi'], name=full_param_name))

    # 2. 목적 함수(Objective Function) 정의
    #    이 함수는 skopt가 주는 파라미터로 오디오를 처리하고 점수를 반환합니다.
    @use_named_args(search_space)
    def vanilla_objective_function(**params):
        temp_config = [p.copy() for p in initial_config]
        active_config = []

        for fx in temp_config:
            fx_type = fx['type']
            
            # EQ는 항상 활성화
            is_active = (fx_type == "EQ") or (params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                # 활성화된 이펙터의 파라미터 값 업데이트
                for p_name, p_value in params.items():
                    p_fx_type, param_key = p_name.split('__')
                    if p_fx_type == fx_type and param_key != 'activation':
                        keys = param_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in fx: fx[keys[0]] = {}
                            fx[keys[0]][keys[1]] = p_value
                        else:
                            fx[param_key] = p_value
                active_config.append(fx)

        processed_audio = render(audio, sr, active_config)
        score = clap_score(processed_audio, sr, prompt, model, processor, device)
        return -1.0 * score

    scores_history = {}
    
    @use_named_args(search_space)
    def objective_function_negative(**params):
        temp_config = [p.copy() for p in initial_config]
        active_config = []

        for fx in temp_config:
            fx_type = fx['type']
            
            # EQ는 항상 활성화
            is_active = (fx_type == "EQ") or (params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                # 활성화된 이펙터의 파라미터 값 업데이트
                for p_name, p_value in params.items():
                    p_fx_type, param_key = p_name.split('__')
                    if p_fx_type == fx_type and param_key != 'activation':
                        keys = param_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in fx: fx[keys[0]] = {}
                            fx[keys[0]][keys[1]] = p_value
                        else:
                            fx[param_key] = p_value
                active_config.append(fx)
        
        processed_audio = render(audio, sr, active_config)
        
        # 긍정 프롬프트와 부정 프롬프트를 모두 사용하여 점수 계산
        positive_prompt = prompt # e.g., "A clear vocal with a subtle club room ambience"
        negative_prompt = "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound"
        
        # 두 개의 프롬프트에 대한 점수를 배치로 한 번에 계산
        prompts = [positive_prompt, negative_prompt]
        # 오디오는 동일하므로 리스트에 두 번 넣어줌
        audio_batch = [processed_audio, processed_audio]
        
        scores = clap_score_batch_negative(audio_batch, sr, prompts, model, processor, device)
        
        positive_score = scores[0]
        negative_score = scores[1]
        
        # 최종 점수 = 긍정 점수 - 부정 점수
        final_score = positive_score - negative_score
        
        params_tuple = tuple(sorted(params.items()))
        scores_history[params_tuple] = positive_score
        
        # skopt는 최소화를 하므로, 최종 점수에 -1을 곱해 반환
        return -1.0 * final_score
    
    pbar = tqdm(total=n_calls, desc="Bayesian Optimization Progress", unit="iteration")
    def pbar_callback(res):
        pbar.update(1)
    early_stopper = EarlyStopper(delta=0.001, n_best=30)
    # 3. 베이지안 최적화 실행
    objective_function = objective_function_negative if use_negative else vanilla_objective_function
    result = gp_minimize(objective_function, 
                         search_space, 
                         n_calls=n_calls, 
                         acq_func="LCB", 
                         kappa=5, 
                         n_initial_points=20, 
                         random_state=42, 
                         callback=[pbar_callback, early_stopper]
                        )
    pbar.close()
    # 4. 탐색 기록에서 Top-N 결과 추출
    all_results = sorted(zip(result.func_vals, result.x_iters), key=lambda x: x[0])
    top_n_results = all_results[:top_n]
    final_presets = []
    presets_to_sort = []

    for rank, (score, params_list) in enumerate(top_n_results):
        best_params = dict(zip(param_names, params_list))
        final_config = []
        params_tuple = tuple(sorted(best_params.items()))
        benchmark_clap_score = scores_history.get(params_tuple, 0.0) # 만약의 경우를 대비해 기본값 0.0

        for fx in initial_config:
            fx_type = fx['type']
            
            is_active = (fx_type == "EQ") or (best_params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                new_fx = fx.copy()
                for p_name, p_value in best_params.items():
                    p_fx_type, p_key = p_name.split('__')
                    if p_fx_type == fx_type and p_key != 'activation':
                        keys = p_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in new_fx: new_fx[keys[0]] = {}
                            new_fx[keys[0]][keys[1]] = p_value
                        else:
                            new_fx[p_key] = p_value
                final_config.append(new_fx)
                
        presets_to_sort.append({
            "composite_score": -1.0 * score,
            "benchmark_clap_score": benchmark_clap_score,
            "plugins": final_config
        })
        
        # --- 5. Benchmark CLAP Score 기준으로 재정렬 및 파일 저장 ---
        # benchmark_clap_score가 높은 순으로 리스트를 다시 정렬합니다.
        final_presets = sorted(presets_to_sort, key=lambda x: x['benchmark_clap_score'], reverse=True)

        for rank, preset in enumerate(final_presets):
            # 렌더링은 저장 직전에 한 번만 수행
            final_audio = render(audio, sr, preset['plugins'])
            
            # 재정렬된 순서에 따라 파일 이름 결정
            if rank == 0:
                filename = "best.wav"
            else:
                filename = f"rank_{rank + 1}.wav"
                
            save_audio(os.path.join(outdir, filename), final_audio, sr)
            
            # 최종 순위를 딕셔너리에 추가
            preset['rank'] = rank + 1
    return final_presets

PARAM_RANGES = {
    "Distortion": {"drive_db": {"lo": 0, "hi": 15, "res": 0.1, "scale": "linear"}},
    "EQ": {
        "mode": {"choices": ["pass-pass", "pass-shelf", "shelf-pass", "shelf-shelf"], "type": "categorical"},
        "low_cut": {"lo": 50, "hi": 500, "res": 10, "scale": "log"},
        "high_cut": {"lo": 8000, "hi": 16000, "res": 100, "scale": "log"},
        "q": {"lo": 0.1, "hi": 10.0, "res": 0.1, "scale": "linear"},
        "gains.low_shelf": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.high_shelf": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak1": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak2": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak3": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "peak1_freq": {"lo": 100.0, "hi": 500.0, "res": 10.0, "scale": "log"},
        "peak2_freq": {"lo": 500.0, "hi": 4000.0, "res": 100.0, "scale": "log"},
        "peak3_freq": {"lo": 4000.0, "hi": 12000.0, "res": 1000.0, "scale": "log"}
    },
    "Reverb": {
        "room_size": {"lo": 0.0, "hi": 1.0, "res": 0.05, "scale": "linear"},
        "damping": {"lo": 0.0, "hi": 1.0, "res": 0.05, "scale": "linear"},
        "wet_level": {"lo": 0.00, "hi": 1.0, "res": 0.01, "scale": "linear"},
    },
    "Delay": {"delay": {"lo": 0.0, "hi": 0.05, "res": 0.01, "scale": "linear"}},
    "PitchShift": {"semitones": {"lo": -12, "hi": 12, "res": 1, "scale": "linear"}},
    "Bitcrush": {"bit_depth": {"lo": 0, "hi": 16, "res": 1, "scale": "linear"}},
}

# ==============================================================================
# Main Logic
# ==============================================================================
def main():
    from datetime import datetime
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="laion/clap-htsat-unfused")
    ap.add_argument("--top_n", type=int, default=5, help="Number of top candidates to refine in parallel.")
    ap.add_argument("--n_calls", type=int, default=100, help="Number of calls for Bayesian optimization.")
    ap.add_argument("--use_negative", action="store_true", default=True, help="Whether to use negative prompts.")
    args = ap.parse_args()
    prompt_folder = safe_folder_name(args.prompt)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"{prompt_folder}_{run_time}")
    os.makedirs(outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    audio, sr = load_audio_mono(args.audio)

    # 원본 오디오 저장
    save_audio(os.path.join(outdir, "original.wav"), audio, sr)

    start_time = time.time()

    initial_config = [
        {"type": "EQ", "mode": "shelf", "low_cut": 120.0, "high_cut": 12000.0, "q": 1.0, "gains": {}, 
         "peak1_freq": 200.0, "peak2_freq": 1000.0, "peak3_freq": 5000.0},
        {"type": "Distortion", "drive_db": 1.0},
        {"type": "Reverb", "room_size": 0.3, "damping": 0.5, "wet_level": 0.1},
        {"type": "Delay", "delay": 0.1},
        {"type": "PitchShift", "semitones": 0},
        {"type": "Bitcrush", "bit_depth": 0},
    ]

    # --- Step 2: 단일 후보군에 대해 베이지안 최적화 실행 ---
    # 병렬 처리가 필요 없으므로 ProcessPoolExecutor 제거
    print(f"\n--- Starting Bayesian Optimization for all parameters ---")
    
    # refine_candidate 함수에 필요한 모든 인자를 넘겨줍니다.
    args_dict = {
        'initial_config': initial_config,
        'audio': audio, 'sr': sr, 'PARAM_RANGES': PARAM_RANGES,
        'model_name': args.model, 'prompt': args.prompt, 'outdir': outdir,
        'top_n': args.top_n, 'n_calls': args.n_calls, 'use_negative': args.use_negative
    }

    final_result = refine_candidate_bayesian(args_dict)

    # --- Step 3: Save final presets ---
    elapsed = time.time() - start_time
    print(f"\nRefinement finished. Total search time: {elapsed:.2f} seconds")
    
    final_output = {
        "prompt": args.prompt,
        "search_time_seconds": elapsed,
        "results": final_result
    }
    with open(os.path.join(outdir, "best_presets.json"), "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Top {len(final_result)} configs saved to best_presets.json at {outdir}.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()