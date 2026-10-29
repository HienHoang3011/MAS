# MAS — agent trò chuyện và hướng dẫn học lý thuyết đồ thị

API FastAPI gồm các agent định tuyến, trò chuyện/gia sư, truy xuất giáo trình, kiểm tra câu trả lời và tóm tắt bộ nhớ. Luồng mới dùng Qwen qua proxy tương thích OpenAI; tiếp tục hỗ trợ cấu hình Gemini/OpenAI hiện có.

- **Trò chuyện:** phản hồi tự nhiên, sử dụng ngữ cảnh hội thoại.
- **Hỏi bài:** gợi ý cách tiếp cận, giải thích phương pháp và hỏi bước tiếp theo; không đưa đáp án cuối, lời giải hoàn chỉnh hay xác nhận một đáp án được đề xuất.
- **Hỏi lý thuyết:** dùng BGE-M3 để lấy các đoạn liên quan trong giáo trình MongoDB, giải thích khái niệm và dẫn nguồn.
- **Review:** Replies are generated as plain text. A source-blind check reviews completeness; theory replies also receive a brief factual assessment per text block. Tutoring is checked for a useful method and next action, followed by a separate source and tutoring-policy audit. Coverage and factual checks reference indexed text blocks verified by the application. A rejected draft is rewritten once with its feedback; only an accepted reply is saved and returned. Exhausted checks return HTTP 502. These remain LLM judgments, not a correctness guarantee.

## Chạy tại máy

Yêu cầu Python 3.12+ và `uv`.

```bash
uv sync --frozen --extra dev
cp .env.example .env
# Điền cấu hình proxy và MongoDB trong .env.
make index-knowledge
make run
```

Nếu đã có `.env`, giữ cấu hình hiện có. `.env` được Git bỏ qua; không đưa credentials vào tài liệu hoặc mã nguồn. Swagger ở `http://127.0.0.1:2003/docs`.

Các biến chính:

| Biến | Vai trò |
| --- | --- |
| `LLM_PROVIDER=proxy` | Dùng proxy Qwen |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT` | Kết nối proxy; URL kết thúc bằng `/v1` |
| `MONGODB_URI` | Kết nối MongoDB, cần hỗ trợ transaction (Atlas/replica set) |
| `MONGODB_DATABASE=mas` | Lưu phiên, tin nhắn, tóm tắt và embedding |
| `MONGODB_KNOWLEDGE_DATABASE=data` | Database giáo trình |
| `MONGODB_KNOWLEDGE_COLLECTION=gtrinh` | Collection chứa các đoạn giáo trình |
| `EMBEDDING_MODEL=BAAI/bge-m3` | Mô hình embedding |
| `EMBEDDING_CACHE_DIR=.cache/models` | Cache mô hình trong dự án |
| `SUMMARY_POLL_SECONDS=5` | Chu kỳ kiểm tra hàng đợi tóm tắt |

`make index-knowledge` đọc các trường `chunk_id`, `source`, `heading_path`, `content` trong `data.gtrinh`. Lệnh tải BGE-M3 ở lần đầu, tạo vector dense 1024 chiều và lưu vào `mas.knowledge_embeddings`; giữ nguyên dữ liệu nguồn. Chạy lại sẽ bỏ qua các đoạn có cùng nội dung và mô hình. Với corpus nhỏ, truy xuất tính cosine trên các vector MongoDB, lấy tối đa 4 đoạn; không cần cấu hình Atlas Vector Search. Corpus lớn hơn cần chuyển bước tìm vector sang chỉ mục chuyên dụng.

## Chat API

```bash
curl -X POST http://127.0.0.1:2003/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"content":"Hướng dẫn mình tìm đường đi ngắn nhất bằng Dijkstra."}'
```

Input gồm `content`, `user_id` tùy chọn và `session_id` tùy chọn:

- Không có `user_id`: tạo người dùng khách và phiên mới; trả cả hai ID trong response.
- Có `user_id`, không có `session_id`: tiếp tục phiên gần nhất của người dùng; tạo mới nếu chưa có.
- Có cả hai ID: tiếp tục đúng phiên; trả 404 nếu phiên không thuộc người dùng.
- Muốn mở phiên mới cho người dùng đã có: `POST /v1/chat/sessions` với `{"user_id":"student-01"}`.

Response có `user_id`, `session_id`, `route` (`conversation`, `tutoring`, `knowledge`), `answer`, `sources`, `summary_pending`. `sources` chỉ chứa ID, tên nguồn và điểm tương đồng; không trả nguyên văn đoạn giáo trình có thể chứa đáp án.

Tiếp tục phiên bằng hai ID đã nhận:

```bash
curl -X POST http://127.0.0.1:2003/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"content":"Vì sao cần kiểm tra trọng số không âm?","user_id":"<returned-user-id>","session_id":"<returned-session-id>"}'
```

## Bộ nhớ

Mỗi lượt thành công lưu nguyên tử hai tin (user, assistant) và số thứ tự trong MongoDB. Nếu hai request cùng sửa một phiên, request dùng ngữ cảnh cũ nhận HTTP 409 và cần gửi lại. Request lỗi không lưu một lượt dở dang.

**Short-term:** luôn lấy tối đa **10 tin nhắn gần nhất** đã lưu, theo thứ tự thời gian (10 tin, không phải 10 lượt). Tin hiện tại được gửi riêng cho agent. Agent cũng nhận tối đa 5 bản tóm tắt gần nhất của đúng phiên.

**Long-term:** worker nền tóm tắt các tin nằm ngoài 10 tin gần nhất, tối đa 20 tin mỗi batch. Tóm tắt lưu trong `chat_memories`, kèm `user_id`, `session_id`, khoảng số thứ tự và khoảng thời gian. Toàn bộ tin gốc vẫn được giữ trong `chat_messages`. Trong lúc worker chưa hoàn tất, `summary_pending=true`; những tin cũ chưa được tóm tắt sẽ chưa có trong ngữ cảnh LLM.

Job được xác định từ trạng thái phiên trong MongoDB và có lease để các worker không xử lý trùng. Khởi động lại sẽ tiếp tục công việc chưa xong; nếu tiến trình bị dừng đột ngột, lease hết hạn sau tối đa 16 phút. Lỗi tạm thời được retry có giới hạn, exponential backoff và jitter; lỗi cố định dừng xử lý với `summary_failures=3`. Tin gốc không bị xóa.

Các endpoint đọc đều yêu cầu `user_id` và kiểm tra quan hệ với phiên:

| Endpoint | Chức năng |
| --- | --- |
| `GET /v1/chat/sessions/{session_id}` | Trạng thái phiên và tiến độ tóm tắt |
| `GET /v1/chat/sessions/{session_id}/messages` | Lịch sử, lọc bằng `since`, `until`, `after_sequence`, `limit` |
| `GET /v1/chat/sessions/{session_id}/memories` | Tóm tắt, lọc bằng `since`, `until`, `limit` |

`since`/`until` dùng ISO 8601 có múi giờ, ví dụ `2026-10-23T00:00:00Z`. Với tóm tắt, trả những khoảng thời gian giao với khoảng truy vấn. Lịch sử tin nhắn phân trang tăng dần theo `after_sequence`; tóm tắt lấy các bản mới nhất trong khoảng rồi trả theo thứ tự cũ đến mới.

`user_id` hiện là khóa phân vùng dữ liệu, chưa phải xác thực danh tính. Dự án vẫn cần cơ chế xác thực thật trước khi mở API công khai.

API cũ `/v1/agent/sessions` dùng `problem`/`context` vẫn được giữ để tương thích; dùng `/v1/chat` cho luồng gia sư và bộ nhớ mới.

## Cấu trúc

| File | Vai trò |
| --- | --- |
| `app/api/routers/chat.py` | API chat, phiên và bộ nhớ |
| `app/core/models/chat.py` | Schema đầu vào, định tuyến, tin nhắn và tóm tắt |
| `app/core/use_cases/chat_system.py` | Điều phối các agent và worker tóm tắt |
| `app/infra/db/chat_store.py` | MongoDB, transaction, phân vùng theo phiên và lease |
| `app/services/llm_service.py` | Adapter LLM và kiểm tra JSON response |
| `app/services/embedding_service.py` | Embedding BGE-M3 chạy ngoài event loop |
| `app/services/vector_store.py` | Index và truy xuất giáo trình bằng cosine |
| `scripts/index_knowledge.py` | Index tăng dần, chạy lại được |

## Kiểm thử

```bash
make test
make lint
make type
# Opt-in: MongoDB thật từ .env, LLM giả; tạo và xóa database test riêng.
MAS_LIVE_MONGO=1 uv run --extra dev pytest tests/integration/test_chat_mongo.py
```

Bộ test mặc định không gọi LLM hay MongoDB thật. Test MongoDB opt-in kiểm tra lưu nguyên tử, giới hạn 10 tin, tóm tắt, lọc thời gian và phân vùng phiên/người dùng.
