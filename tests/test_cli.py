from settlement_tool.cli import build_parser, build_ocr_options


def test_cli_accepts_chandra_backend_options():
    parser = build_parser()

    args = parser.parse_args(
        [
            "analyze",
            "--ocr-backend",
            "chandra",
            "--hf-provider",
            "auto",
            "--hf-model",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--hf-max-tokens",
            "128",
            "--hf-timeout-seconds",
            "60",
            "--hf-image-max-side",
            "1200",
            "--hf-token-env",
            "MY_HF_TOKEN",
            "--hf-base-url",
            "https://example.endpoints.huggingface.cloud/v1",
            "--hf-bill-to",
            "my-org",
            "--chandra-method",
            "hf",
            "--chandra-prompt-type",
            "ocr_layout",
            "--chandra-model-checkpoint",
            "models/chandra-ocr-2",
            "--chandra-torch-device",
            "cpu",
            "--chandra-max-output-tokens",
            "256",
            "--chandra-image-max-side",
            "1400",
            "--chandra-prompt",
            "account only",
        ]
    )
    options = build_ocr_options(args)

    assert options.backend == "chandra"
    assert options.hf_provider == "auto"
    assert options.hf_model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert options.hf_max_tokens == 128
    assert options.hf_timeout_seconds == 60
    assert options.hf_image_max_side == 1200
    assert options.hf_token_env == "MY_HF_TOKEN"
    assert options.hf_base_url == "https://example.endpoints.huggingface.cloud/v1"
    assert options.hf_bill_to == "my-org"
    assert options.chandra_method == "hf"
    assert options.chandra_prompt_type == "ocr_layout"
    assert options.chandra_model_checkpoint == "models/chandra-ocr-2"
    assert options.chandra_torch_device == "cpu"
    assert options.chandra_max_output_tokens == 256
    assert options.chandra_image_max_side == 1400
    assert options.chandra_prompt == "account only"
