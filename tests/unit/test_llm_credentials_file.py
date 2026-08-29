import pytest

from remcts.core.state import SearchConfig
from remcts.llm.client import LLMClientError, OpenAICompatibleLLMClient, create_llm_client


def test_create_llm_client_loads_key_value_credentials(tmp_path):
    credentials_file = tmp_path / "credentials.yaml"
    credentials_file.write_text(
        "\n".join(
            [
                "api_key: sk-test-key",
                "base_url: https://example.test/v1",
                "model_name: gpt-test",
            ]
        ),
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_key == "sk-test-key"
    assert client.base_url == "https://example.test/v1"
    assert client.model_name == "gpt-test"
    assert client.api_alias is None


def test_create_llm_client_loads_plain_credentials(tmp_path):
    credentials_file = tmp_path / "credentials.txt"
    credentials_file.write_text(
        "sk-plain-key\nhttps://plain.example.test/v1\ngpt-plain\n",
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_key == "sk-plain-key"
    assert client.base_url == "https://plain.example.test/v1"
    assert client.model_name == "gpt-plain"


def test_create_llm_client_loads_mixed_plain_and_key_value_credentials(tmp_path):
    credentials_file = tmp_path / "credentials.txt"
    credentials_file.write_text(
        "sk-mixed-key\nbase_url: https://mixed.example.test/v1\nmodel_name: gpt-mixed\n",
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_key == "sk-mixed-key"
    assert client.base_url == "https://mixed.example.test/v1"
    assert client.model_name == "gpt-mixed"


def test_create_llm_client_falls_back_to_env_when_file_missing(monkeypatch):
    monkeypatch.setenv("REMCTS_TEST_API_KEY", "sk-env-key")
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_API_KEY",
        llm_credentials_file="missing_credentials_file.txt",
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_key == "sk-env-key"
    assert client.base_url == "https://default.test/v1"
    assert client.model_name == "gpt-default"


def test_create_llm_client_selects_json_api_alias(tmp_path):
    credentials_file = tmp_path / "credentials.json"
    credentials_file.write_text(
        """
{
  "apis": [
    {
      "alias": "ballbox",
      "api_key": "sk-ballbox",
      "base_url": "https://ballbox.example.test/v1",
      "model_name": "gpt-ballbox"
    },
    {
      "alias": "ark",
      "api_key": "sk-ark",
      "base_url": "https://ark.example.test/api/v3",
      "model_name": "ep-ark"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
        llm_api_alias="ark",
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_alias == "ark"
    assert client.api_key == "sk-ark"
    assert client.base_url == "https://ark.example.test/api/v3"
    assert client.model_name == "ep-ark"


def test_qwen3_non_streaming_disables_thinking_by_default(monkeypatch):
    client = OpenAICompatibleLLMClient(
        model_name="qwen3-1.7b",
        base_url="https://dashscope.example.test/compatible-mode/v1",
        api_key="sk-test",
    )
    captured = {}

    def fake_chat_completion(payload):
        captured.update(payload)
        return "ok"

    monkeypatch.setattr(client, "_chat_completion", fake_chat_completion)

    assert client.generate("hello") == "ok"
    assert captured["enable_thinking"] is False


def test_gpt5_disable_thinking_uses_reasoning_effort_none(monkeypatch):
    client = OpenAICompatibleLLMClient(
        model_name="gpt-5.2",
        base_url="https://example.test/v1",
        api_key="sk-test",
        enable_thinking=False,
    )
    captured = {}

    def fake_chat_completion(payload):
        captured.update(payload)
        return "ok"

    monkeypatch.setattr(client, "_chat_completion", fake_chat_completion)

    assert client.generate("hello") == "ok"
    assert captured["reasoning_effort"] == "none"
    assert "enable_thinking" not in captured


def test_gpt5_enable_thinking_omits_reasoning_override(monkeypatch):
    client = OpenAICompatibleLLMClient(
        model_name="gpt-5.2",
        base_url="https://example.test/v1",
        api_key="sk-test",
        enable_thinking=True,
    )
    captured = {}

    def fake_chat_completion(payload):
        captured.update(payload)
        return "ok"

    monkeypatch.setattr(client, "_chat_completion", fake_chat_completion)

    assert client.generate("hello") == "ok"
    assert "reasoning_effort" not in captured
    assert "enable_thinking" not in captured


def test_create_llm_client_selects_compact_json_array_api_alias(tmp_path):
    credentials_file = tmp_path / "credentials.json"
    credentials_file.write_text(
        '[{"alias":"ballbox","api_key":"sk-ballbox"},{"alias":"ark","api_key":"sk-ark","base_url":"https://ark.example.test/api/v3","model_name":"ep-ark"}]',
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
        llm_api_alias="ark",
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_alias == "ark"
    assert client.api_key == "sk-ark"
    assert client.base_url == "https://ark.example.test/api/v3"
    assert client.model_name == "ep-ark"


def test_create_llm_client_selects_text_api_alias(tmp_path):
    credentials_file = tmp_path / "credentials.txt"
    credentials_file.write_text(
        """
alias: ballbox
api_key: sk-ballbox
base_url: https://ballbox.example.test/v1
model_name: gpt-ballbox

alias: ark
api_key: sk-ark
base_url: https://ark.example.test/api/v3
model_name: ep-ark
""".strip(),
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
        llm_api_alias="ballbox",
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_alias == "ballbox"
    assert client.api_key == "sk-ballbox"
    assert client.base_url == "https://ballbox.example.test/v1"
    assert client.model_name == "gpt-ballbox"


def test_create_llm_client_selects_sectioned_api_alias(tmp_path):
    credentials_file = tmp_path / "credentials.txt"
    credentials_file.write_text(
        """
[ballbox]
api_key: sk-ballbox
base_url: https://ballbox.example.test/v1
model_name: gpt-ballbox

[ark]
api_key: sk-ark
base_url: https://ark.example.test/api/v3
model_name: ep-ark
""".strip(),
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
        llm_api_alias="ark",
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_alias == "ark"
    assert client.api_key == "sk-ark"
    assert client.base_url == "https://ark.example.test/api/v3"
    assert client.model_name == "ep-ark"


def test_create_llm_client_uses_default_alias_when_config_alias_missing(tmp_path):
    credentials_file = tmp_path / "credentials.txt"
    credentials_file.write_text(
        """
default_alias: ark

alias: ballbox
api_key: sk-ballbox
base_url: https://ballbox.example.test/v1
model_name: gpt-ballbox

alias: ark
api_key: sk-ark
base_url: https://ark.example.test/api/v3
model_name: ep-ark
""".strip(),
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.api_alias == "ark"
    assert client.api_key == "sk-ark"
    assert client.base_url == "https://ark.example.test/api/v3"
    assert client.model_name == "ep-ark"


def test_create_llm_client_requires_alias_for_multiple_profiles(tmp_path):
    credentials_file = tmp_path / "credentials.txt"
    credentials_file.write_text(
        """
alias: ballbox
api_key: sk-ballbox

alias: ark
api_key: sk-ark
""".strip(),
        encoding="utf-8",
    )
    config = SearchConfig(
        llm_model_name="gpt-default",
        llm_base_url="https://default.test/v1",
        llm_api_key_env="REMCTS_TEST_MISSING",
        llm_credentials_file=str(credentials_file),
    )

    with pytest.raises(LLMClientError, match="Multiple API profiles"):
        create_llm_client(config)
