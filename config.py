# config.py
# ---------------------------------------------
# Konfigurasi provider & model untuk aplikasi RAG.
# - Tidak bergantung pada Streamlit (UI akan di app.py)
# - Menyediakan utilitas untuk:
#   * daftar provider & model
#   * membaca API keys dari environment
#   * validasi konfigurasi
#   * inisialisasi klien LLM (OpenAI / Gemini / DeepSeek / OpenRouter)
#   * simpan konfigurasi run ke YAML (opsional)
# ---------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# LangChain chat models (gunakan hanya saat diperlukan)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# =========================
# 1) Konstanta & Model Map
# =========================

PROVIDERS = [
    "OpenAI",
    "Gemini",
    "DeepSeek",
    "OpenRouter (OpenAI-compatible)",
    "Non-LLM (hanya retrieval)",
]

MODEL_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "OpenAI": ("gpt-4o-mini", "gpt-4o"),
    "Gemini": ("gemini-1.5-flash", "gemini-1.5-pro"),
    "DeepSeek": ("deepseek-chat", "deepseek-reasoner"),
    "OpenRouter (OpenAI-compatible)": (
        "openrouter/deepseek/deepseek-chat",
        "openrouter/google/gemini-1.5-flash",
        "openrouter/openai/gpt-4o-mini",
    ),
    "Non-LLM (hanya retrieval)": ("-",),
}

# Nama variabel environment yang didukung
ENV_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Endpoint khusus (hanya saat perlu override)
DEFAULT_ENDPOINTS = {
    "DeepSeek": "https://api.deepseek.com",           # OpenAI-compatible
    "OpenRouter (OpenAI-compatible)": "https://openrouter.ai/api/v1",
}


# =========================
# 2) DataClass Konfigurasi
# =========================

@dataclass
class LLMKeys:
    openai: str = ""
    google: str = ""
    deepseek: str = ""
    openrouter: str = ""

    @classmethod
    def from_env(cls) -> "LLMKeys":
        """Ambil keys dari environment variables (jika ada)."""
        # Load .env file if present
        load_dotenv()

        return cls(
            openai=os.getenv(ENV_KEY_MAP["openai"], "") or "",
            google=os.getenv(ENV_KEY_MAP["google"], "") or "",
            deepseek=os.getenv(ENV_KEY_MAP["deepseek"], "") or "",
            openrouter=os.getenv(ENV_KEY_MAP["openrouter"], "") or "",
        )

    def merged_with(self, overrides: Optional[Dict[str, str]]) -> "LLMKeys":
        """Gabungkan dengan override (mis. dari UI) tanpa menimpa nilai kosong."""
        if not overrides:
            return self
        data = asdict(self)
        for k, v in (overrides or {}).items():
            if v:  # hanya timpa jika ada nilai
                data[k] = v
        return LLMKeys(**data)


@dataclass
class AppConfig:
    provider: str
    model: str
    temperature: float = 0.2
    # biarkan None jika tidak ingin di-set
    max_output_tokens: Optional[int] = None
    timeout: Optional[float] = 120.0         # detik, untuk panggilan LLM
    # catatan lain / metadata run
    project_name: str = "default_corpus"
    persist_dir: str = "rag_cache"

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Provider '{self.provider}' tidak dikenal. Pilihan: {PROVIDERS}")
        allowed = MODEL_OPTIONS.get(self.provider, ())
        if self.model not in allowed:
            raise ValueError(
                f"Model '{self.model}' tidak valid untuk provider '{self.provider}'. "
                f"Allowed: {allowed}"
            )
        if self.provider == "Non-LLM (hanya retrieval)" and self.model != "-":
            raise ValueError("Untuk provider Non-LLM, model harus '-'.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature harus di rentang [0.0, 2.0].")


# =========================
# 3) Utilitas Konfigurasi
# =========================

def get_models_for_provider(provider: str) -> Tuple[str, ...]:
    """Daftar model yang tersedia untuk provider tertentu."""
    return MODEL_OPTIONS.get(provider, ("-",))


def _require_key(keys: LLMKeys, name: str) -> str:
    """Ambil key dari LLMKeys; error jika kosong."""
    value = getattr(keys, name, "")
    if not value:
        env_var = ENV_KEY_MAP.get(name, name.upper())
        raise ValueError(
            f"API key untuk '{name}' kosong. Set melalui environment '{env_var}' "
            f"atau berikan override saat runtime."
        )
    return value


def ensure_llm(config: AppConfig, keys: Optional[LLMKeys] = None) -> Optional[Any]:
    """
    Buat instance Chat model sesuai provider. Kembalikan None bila Non-LLM.
    Catatan:
    - Tidak ada dependensi ke Streamlit; error dilempar sebagai ValueError.
    - Untuk OpenRouter/DeepSeek, gunakan ChatOpenAI dengan base_url kompatibel.
    """
    config.validate()
    keys = keys or LLMKeys.from_env()

    if config.provider == "Non-LLM (hanya retrieval)":
        return None

    # Parameter umum
    common_kwargs = {
        "temperature": config.temperature,
        # sebagian wrapper tidak menerima max_tokens; kita pasang kondisional
    }

    if config.max_output_tokens is not None:
        # LangChain ChatOpenAI menggunakan 'max_tokens'
        common_kwargs["max_tokens"] = int(config.max_output_tokens)

    # OpenAI
    if config.provider == "OpenAI":
        api_key = _require_key(keys, "openai")
        # untuk downstream libs yang cek ENV
        os.environ[ENV_KEY_MAP["openai"]] = api_key
        return ChatOpenAI(model=config.model, **common_kwargs)

    # Gemini
    if config.provider == "Gemini":
        api_key = _require_key(keys, "google")
        os.environ[ENV_KEY_MAP["google"]] = api_key
        # ChatGoogleGenerativeAI menerima 'max_output_tokens' bukan 'max_tokens'
        g_kwargs = dict(common_kwargs)
        if "max_tokens" in g_kwargs:
            g_kwargs["max_output_tokens"] = g_kwargs.pop("max_tokens")
        return ChatGoogleGenerativeAI(model=config.model, **g_kwargs)

    # DeepSeek (OpenAI-compatible endpoint)
    if config.provider == "DeepSeek":
        api_key = _require_key(keys, "deepseek")
        base_url = DEFAULT_ENDPOINTS["DeepSeek"]
        return ChatOpenAI(model=config.model, api_key=api_key, base_url=base_url, **common_kwargs)

    # OpenRouter (OpenAI-compatible endpoint)
    if config.provider == "OpenRouter (OpenAI-compatible)":
        api_key = _require_key(keys, "openrouter")
        base_url = DEFAULT_ENDPOINTS["OpenRouter (OpenAI-compatible)"]
        return ChatOpenAI(model=config.model, api_key=api_key, base_url=base_url, **common_kwargs)

    # Should not reach here due to validate()
    raise ValueError(f"Provider tidak didukung: {config.provider}")


# =========================
# 4) Persist Konfigurasi
# =========================

def dump_run_config_yaml(config: AppConfig, keys: Optional[LLMKeys], path: str | Path) -> Path:
    """
    Simpan konfigurasi run ke YAML (tanpa menyimpan API key mentah).
    Hanya simpan info provider/model/temperature/dll & info apakah key tersedia.
    """
    import yaml  # lazy import agar dependency ini opsional

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe = {
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "timeout": config.timeout,
        "project_name": config.project_name,
        "persist_dir": config.persist_dir,
        "keys_available": {
            "openai": bool(keys and keys.openai),
            "google": bool(keys and keys.google),
            "deepseek": bool(keys and keys.deepseek),
            "openrouter": bool(keys and keys.openrouter),
        },
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(safe, f, allow_unicode=True, sort_keys=False)
    return path


# =========================
# 5) Helper
# =========================

def summarize_llm_choice(config: AppConfig) -> str:
    """Ringkasan singkat pilihan LLM untuk ditampilkan di UI/log."""
    if config.provider == "Non-LLM (hanya retrieval)":
        return "Mode Non-LLM (retrieval only)"
    return f"{config.provider} · {config.model} · temp={config.temperature}"
