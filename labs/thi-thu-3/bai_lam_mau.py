"""Khung bài làm Vòng 2 (Coding interview mô phỏng), Đề thi thử 3.

Chép file này thành `bai_lam.py` ở thư mục gốc của sách rồi điền hai bài trong 70 phút của Vòng 2.
Chỉ dùng thư viện chuẩn của Python và NumPy; không cài hay import thư viện ngoài.
Được chạy thử với dữ liệu tự nghĩ ra (viết ở khối `if __name__ == "__main__":` bên dưới).

Sau khi hết giờ mới chấm (đừng mở bộ chấm trước):
    python labs/thi-thu-3/cham_coding.py --student bai_lam.py
"""


class MetricLog:
    """Sổ ghi một chỉ số (ví dụ loss trên tập dev) của nhiều lần chạy huấn luyện.

    record(run, step, value): ghi giá trị `value` (float) của lần chạy `run` (str) tại bước `step` (int ≥ 0).
        Với mỗi `run`, các lần gọi record có `step` TĂNG NGẶT (lần sau lớn hơn lần trước).
        Các lần chạy khác nhau được ghi xen kẽ nhau.
    at(run, step): trả về `value` của bản ghi có step lớn nhất mà vẫn ≤ `step` (int ≥ 0) trong lần chạy `run`;
        trả về None nếu lần chạy đó chưa có bản ghi nào với step ≤ `step` (kể cả khi `run` chưa từng được ghi).

    Tổng số lần gọi record và at có thể tới 2·10⁵, phần lớn vào cùng một lần chạy; bộ chấm có giới hạn
    thời gian cho cả chuỗi lời gọi."""

    def __init__(self):
        # TODO
        raise NotImplementedError

    def record(self, run, step, value):
        # TODO
        raise NotImplementedError

    def at(self, run, step):
        # TODO
        raise NotImplementedError


def corpus_bleu(hyps, refs, max_n=4):
    """BLEU cấp corpus, mỗi câu MỘT câu tham chiếu, không làm trơn.

    hyps: list các câu máy dịch; mỗi câu là list các token (str), có thể rỗng.
    refs: list cùng độ dài với hyps; refs[i] là list token của câu tham chiếu cho hyps[i].
    Trả về (bleu, precisions, bp):
      precisions: list gồm max_n số thực; precisions[n − 1] là precision n-gram đã cắt của cả corpus:
          tử số = tổng trên mọi câu i, trên mọi n-gram g của hyps[i], của min(số lần g có trong hyps[i],
                  số lần g có trong refs[i]);
          mẫu số = tổng số n-gram của mọi hyps[i] (một câu dài L có max(0, L − n + 1) n-gram).
          Nếu mẫu số bằng 0 thì precision đó là 0.0.
      bp: brevity penalty với c = tổng độ dài mọi hyps[i], r = tổng độ dài mọi refs[i]:
          bp = 1.0 nếu c > r; bp = exp(1 − r/c) nếu 0 < c ≤ r; bp = 0.0 nếu c = 0.
      bleu = bp · exp((1/max_n) · Σₙ ln precisions[n − 1]); nếu có precision bằng 0 thì bleu = 0.0.
    bleu, bp và mọi phần tử của precisions là float."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Chỗ để bạn tự thử với dữ liệu nhỏ. Hai ví dụ của đề:
    # log = MetricLog(); log.record("a", 10, 0.9); log.record("a", 20, 0.8)
    # print(log.at("a", 15), log.at("a", 20), log.at("a", 5))          # cần: 0.9 0.8 None
    # print(corpus_bleu([["the", "cat", "sat", "on"]], [["the", "cat", "sat", "on", "the", "mat"]], max_n=2))
    #                                                                  # cần: bleu ≈ 0.6065, precisions [1.0, 1.0], bp ≈ 0.6065
    pass
