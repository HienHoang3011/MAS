# Bộ câu hỏi thử API lý thuyết đồ thị

Gửi từng câu vào trường `content` của `POST /v1/chat` sau khi đã đánh chỉ mục giáo trình. Cột **route dự kiến** giúp kiểm tra định tuyến, không phải kết quả được mã hóa cứng. Với câu lý thuyết, xem `answer` có giải thích đúng và `sources` có nguồn phù hợp. Với bài tập, agent cần hướng dẫn cách làm và bước tiếp theo, không đưa đáp án cuối.

## Khái niệm và tính chất

| # | Câu hỏi để gửi | Route dự kiến | Điểm cần kiểm tra |
| --- | --- | --- | --- |
| 1 | Đồ thị đơn, đa đồ thị và đồ thị có khuyên khác nhau như thế nào? | `knowledge` | Phân biệt cạnh song song và khuyên. |
| 2 | Bậc của đỉnh trong đồ thị vô hướng được tính thế nào nếu có khuyên? Với đồ thị có hướng, bậc vào và bậc ra là gì? | `knowledge` | Khuyên đóng góp 2 vào bậc vô hướng; phân biệt bậc vào và bậc ra. |
| 3 | Phát biểu định lý bắt tay cho đồ thị vô hướng và giải thích vì sao số đỉnh bậc lẻ luôn chẵn. | `knowledge` | Có phát biểu và giải thích lý do, không chỉ nêu kết luận. |
| 4 | Phân biệt đường đi, đường đi đơn và chu trình trong đồ thị. | `knowledge` | Nêu rõ điều kiện lặp đỉnh/cạnh và điểm đầu, điểm cuối. |
| 5 | Đồ thị có hướng liên thông mạnh khác liên thông yếu như thế nào? | `knowledge` | Giữ đúng hướng của đường đi khi giải thích liên thông mạnh. |
| 6 | Vì sao một cây có `n` đỉnh thì có `n - 1` cạnh? Chỉ biết có `n - 1` cạnh đã đủ kết luận là cây chưa? | `knowledge` | Giải thích vai trò của liên thông và không có chu trình. |
| 7 | Điều kiện nào cho biết một đồ thị vô hướng là đồ thị hai phía? Chu trình độ dài lẻ liên quan gì đến điều kiện đó? | `knowledge` | Nêu quan hệ hai chiều giữa đồ thị hai phía và chu trình lẻ. |
| 8 | Điều kiện tồn tại đường đi Euler và chu trình Euler trong đồ thị vô hướng là gì? Các đỉnh cô lập có ảnh hưởng thế nào? | `knowledge` | Phân biệt đường đi với chu trình; xét tính liên thông của phần có cạnh. |
| 9 | Đường đi Hamilton khác đường đi Euler ở điểm nào? Có thể dùng điều kiện Euler để kết luận Hamilton không? | `knowledge` | Phân biệt đi qua đỉnh với đi qua cạnh. |

## Thuật toán

| # | Câu hỏi để gửi | Route dự kiến | Điểm cần kiểm tra |
| --- | --- | --- | --- |
| 10 | BFS và DFS khác nhau ở cách chọn đỉnh tiếp theo như thế nào? | `knowledge` | Nêu thứ tự duyệt và cấu trúc dữ liệu thường dùng. |
| 11 | Vì sao BFS tìm được đường đi ngắn nhất theo số cạnh trên đồ thị không trọng số? | `knowledge` | Giải thích theo các lớp khoảng cách; không áp dụng nhầm cho trọng số bất kỳ. |
| 12 | Vì sao thuật toán Dijkstra cần trọng số cạnh không âm? | `knowledge` | Giải thích bước chốt khoảng cách và rủi ro khi có cạnh âm. |
| 13 | Cây khung nhỏ nhất khác cây đường đi ngắn nhất từ một đỉnh nguồn như thế nào? | `knowledge` | Phân biệt mục tiêu tổng trọng số với khoảng cách từ nguồn. |
| 14 | Prim và Kruskal xây dựng cây khung nhỏ nhất theo hai cách nào? | `knowledge` | Mô tả cách chọn cạnh và tránh chu trình ở mức khái niệm. |
| 15 | Sắp xếp tô pô là gì và vì sao chỉ áp dụng cho đồ thị có hướng không chu trình? | `knowledge` | Nêu điều kiện DAG và ý nghĩa thứ tự trước/sau. |

## Bài tập và ngữ cảnh nhiều lượt

| # | Câu hỏi để gửi | Route dự kiến | Điểm cần kiểm tra |
| --- | --- | --- | --- |
| 16 | Cho đồ thị vô hướng có các cạnh `AB`, `AC`, `BC`, `CD`. Hãy tính bậc từng đỉnh và xác định đồ thị có đường đi Euler hoặc chu trình Euler không. | `tutoring` | Hướng dẫn đếm bậc và áp dụng điều kiện Euler; không đưa kết quả cuối. |
| 17 | Cho đồ thị có các cạnh trọng số `AB=2`, `AC=5`, `BC=1`, `CD=2`, `BD=6`. Hãy tìm đường đi ngắn nhất từ `A` đến `D` bằng Dijkstra. | `tutoring` | Gợi ý cách cập nhật nhãn khoảng cách; không tính hộ toàn bộ đường đi. |
| 18 | Nếu thêm cạnh `AD` vào đồ thị ở câu 16 thì kết luận về Euler thay đổi như thế nào? | `tutoring` | Gửi bằng **cùng `user_id` và `session_id` của câu 16**; kiểm tra agent dùng đúng đồ thị trước đó và hướng dẫn xét lại bậc. |

Để thử một phiên mới, bỏ `user_id` và `session_id` trong request đầu tiên. Lưu hai ID từ response và gửi lại chúng cho câu 18. Nếu cần mở phiên mới cho cùng người dùng, gọi `POST /v1/chat/sessions`.
