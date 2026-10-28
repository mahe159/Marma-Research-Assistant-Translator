import translate


def test_prompt_template_contains_expected_slots():
    prompt = translate.ALPACA_PROMPT.format(
        source_lang="English", target_lang="Bangla", text="Hello"
    )
    assert "Translate this English sentence into Bangla." in prompt
    assert "### Input:\nHello" in prompt
    assert prompt.rstrip().endswith("### Response:")
