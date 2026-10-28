# API reference

Base URL: `http://localhost:8000` (or your ngrok URL). Interactive docs at `/docs`.

## `GET /`

```json
{"status": "online", "model": "LLaMA-3-8B (Marma Fine-tuned)"}
```

## `POST /translate`

Request body

| Field | Type | Notes |
|---|---|---|
| `text` | string | Required, non-blank |
| `source_lang` | string | Inserted into the prompt as-is |
| `target_lang` | string | Inserted into the prompt as-is |

The frontend sends names with underscores (`Marma_English_Transliterated`); direct callers can use either form, but use the form the model was trained with.

Responses

| Status | Meaning |
|---|---|
| 200 | `{"translated_text", "source_lang", "target_lang"}` |
| 400 | `Text cannot be empty` |
| 422 | Body failed validation |
| 500 | Exception raised during inference |

Language names offered by the frontend: `Bangla`, `English`, `Marma English Transliterated`, `Marma Bangla Transliterated`.
