# MAS — hệ thống agent hỏi, thực thi và kiểm chứng

MAS là API FastAPI triển khai luồng: **phân loại yêu cầu → hỏi khi thiếu thông tin → thực thi → kiểm chứng → tổng hợp**. Hệ thống dùng Gemini hoặc OpenAI để trả lời có cấu trúc và giới hạn số vòng trao đổi.

## Chạy tại máy

Yêu cầu Python 3.12+ và [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/HienHoang3011/MAS.git
cd MAS
uv sync --frozen --extra dev
cp .env.example .env
```

Mở `.env` và chọn một nhà cung cấp:

```dotenv
# Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=<khóa Gemini của bạn>
GEMINI_MODEL=gemini-3.5-flash
```

Hoặc dùng OpenAI với `LLM_PROVIDER=openai`, `OPENAI_API_KEY` và `OPENAI_MODEL`. File `.env` đã được Git bỏ qua; không đưa API key vào `.env.example`.

```bash
make run
```

Server chạy ở `http://127.0.0.1:2003`. Kiểm tra `curl http://127.0.0.1:2003/health/`; giao diện API ở [Swagger UI](http://127.0.0.1:2003/docs).

## Dùng agent API

Tạo phiên:

```bash
curl -X POST http://127.0.0.1:2003/v1/agent/sessions \
  -H 'Content-Type: application/json' \
  -d '{"problem":"Buổi họp diễn ra khi nào?","context":"Buổi họp diễn ra thứ Sáu lúc 10 giờ."}'
```

Phản hồi có `id`, `status`, `question`, `execution`, `verification` và `result`. Nếu `status` là `completed`, đọc `result.answer` và `result.limitations`. Nếu `status` là `waiting_for_user`, trả lời `question.question` bằng `id` của phiên:

```bash
curl -X POST http://127.0.0.1:2003/v1/agent/sessions/<session-id>/replies \
  -H 'Content-Type: application/json' \
  -d '{"answer":"Ngân sách của tôi là 15."}'
```

Xem lại phiên bằng `GET /v1/agent/sessions/<session-id>`. Mặc định hệ thống hỏi tối đa 3 lần và thực thi tối đa 2 lượt; chỉnh qua `AGENT_MAX_CLARIFICATIONS` và `AGENT_MAX_EXECUTIONS` trong `.env`.

## Cấu trúc

| Thư mục | Vai trò |
| --- | --- |
| `app/api/routers/agent.py` | API tạo phiên, trả lời và xem phiên |
| `app/core/models/agent.py` | Trạng thái và dữ liệu trao đổi giữa các agent |
| `app/core/use_cases/agent_system.py` | Điều phối, Socratic, thực thi, kiểm chứng và tổng hợp |
| `app/core/ports/` | Giao diện LLM và công cụ mở rộng |
| `app/services/llm_service.py` | Kết nối Gemini/OpenAI với đầu ra JSON có cấu trúc |
| `app/infra/` | Bộ nhớ phiên và công cụ tìm trong ngữ cảnh |

Công cụ mặc định chỉ tìm trong `context` người dùng cung cấp. Phiên được lưu trong bộ nhớ tiến trình, sẽ mất khi khởi động lại và không dùng chung giữa nhiều worker. API hiện chưa có xác thực thật; chỉ chạy cục bộ cho đến khi bổ sung xác thực trước khi triển khai công khai.

## Kiểm thử

```bash
make test
make lint
```

Bộ test dùng LLM giả, không tốn lượt gọi Gemini/OpenAI. Luồng API đã được thử trực tiếp với Gemini 3.5 Flash bằng câu hỏi số học công khai và trả về HTTP `201`, trạng thái `completed`.
