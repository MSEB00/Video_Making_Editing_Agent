import openai
import pytest

from app.ai.creative_editor import CreativeAIConfigurationError, ShortFormEditingModel
from app.agent.short_form_editor import ShortFormCreativeEditor


def test_gemini_is_default_provider_via_official_compatibility_endpoint(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    options = {}

    class FakeClient:
        pass

    def fake_openai(**kwargs):
        options.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    model = ShortFormEditingModel()

    assert model.provider == "gemini"
    assert model.model == "gemini-3.8-flash"
    assert options["api_key"] == "test-gemini-key"
    assert options["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert options["timeout"] == 90.0


def test_openai_remains_an_explicit_fallback(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    options = {}

    class FakeClient:
        pass

    def fake_openai(**kwargs):
        options.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    model = ShortFormEditingModel()

    assert model.provider == "openai"
    assert model.model == "gpt-4o-mini"
    assert options["api_key"] == "test-openai-key"
    assert "base_url" not in options


def test_gemini_provider_requires_its_own_key(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(CreativeAIConfigurationError, match="GEMINI_API_KEY"):
        ShortFormEditingModel()


def test_editor_keeps_local_fallback_available_without_gemini_key(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    editor = ShortFormCreativeEditor()

    assert editor.model is None
    assert isinstance(editor.model_configuration_error, CreativeAIConfigurationError)