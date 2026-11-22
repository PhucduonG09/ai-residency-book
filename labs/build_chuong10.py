"""Sinh hai notebook cho lab chương 10: lab.ipynb (có TODO) và loigiai.ipynb (lời giải).

Chạy lại script này nếu muốn tạo lại notebook gốc:  python labs/build_chuong10.py
Toàn bộ loigiai.ipynb chạy dưới 2 phút trên CPU (PyTorch CPU, 1 luồng, seed cố định).
Mọi con số của bài săn bug (kể cả tổ hợp bốn lỗi) được tính lại ngay trong notebook.
lab.ipynb không chứa phân tích từng lỗi và không có chú thích chỉ chỗ lỗi; phần đó chỉ nằm trong loigiai.ipynb.
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).parent / "chuong10"
OUT.mkdir(parents=True, exist_ok=True)

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = """import time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.linalg import sqrtm
torch.set_num_threads(1)      # 1 luồng cho kết quả và thời gian ổn định trên CPU ít lõi
np.set_printoptions(precision=4, suppress=True)
T0 = time.time()"""

# ---------------------------------------------------------------- Lab 1
AE_DATA = """def swiss(n, seed):
    \"\"\"Cuộn bánh 3D: một mặt phẳng hai chiều cuộn lại, cộng nhiễu 0.02.\"\"\"
    g = torch.Generator().manual_seed(seed)
    t = (1.5 + 3 * torch.rand(n, generator=g)) * math.pi
    h = torch.rand(n, generator=g) * 2 - 1
    x = torch.stack([t * torch.cos(t) / 10, h, t * torch.sin(t) / 10], 1)
    return x + 0.02 * torch.randn(n, 3, generator=g)

Xtr, Xte = swiss(4000, 0), swiss(2000, 1)
mu0 = Xtr.mean(0)
Xc_tr, Xc_te = Xtr - mu0, Xte - mu0                # đã trừ trung bình của train

def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))

print("tổng phương sai (tập test):", round((Xc_te ** 2).sum(1).mean().item(), 4))"""

AE = ("""def pca_error(k):
    \"\"\"Sai số tái tạo bình phương trung bình (cộng 3 tọa độ) trên Xc_te khi chiếu xuống
    k thành phần chính tính từ Xc_tr (dùng torch.linalg.svd).\"\"\"
    # TODO
    raise NotImplementedError

def train_ae(kind, k, steps=1000):
    \"\"\"kind = 'linear': encoder nn.Linear(3, k, bias=False), decoder nn.Linear(k, 3, bias=False), Adam lr 1e-2.
    kind = 'mlp': encoder mlp(3, k), decoder mlp(k, 3), Adam lr 3e-3.
    Mini-batch 256 lấy ngẫu nhiên từ Xc_tr; loss = tổng bình phương sai số theo 3 tọa độ, trung bình theo mẫu.
    torch.manual_seed(0) ở đầu hàm. Trả về sai số tái tạo trên Xc_te (cùng đơn vị với pca_error).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def pca_error(k):
    _, _, Vt = torch.linalg.svd(Xc_tr, full_matrices=False)
    P = Vt[:k].T @ Vt[:k]                                  # phép chiếu lên k thành phần chính
    return ((Xc_te - Xc_te @ P) ** 2).sum(1).mean().item()

def train_ae(kind, k, steps=1000):
    torch.manual_seed(0)
    if kind == "linear":
        enc, dec, lr = nn.Linear(3, k, bias=False), nn.Linear(k, 3, bias=False), 1e-2
    else:
        enc, dec, lr = mlp(3, k), mlp(k, 3), 3e-3
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr, foreach=True)
    for _ in range(steps):
        x = Xc_tr[torch.randint(0, 4000, (256,))]
        loss = ((dec(enc(x)) - x) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((dec(enc(Xc_te)) - Xc_te) ** 2).sum(1).mean().item()""")

VAE_CODE = ("""def kl_gauss(mu, logvar):
    \"\"\"KL(N(mu, diag(exp(logvar))) ‖ N(0, I)) cộng theo chiều ẩn. Vào (N, k), ra (N,).\"\"\"
    # TODO
    raise NotImplementedError

def reparam(mu, logvar):
    \"\"\"z = mu + σ ⊙ ε với ε ~ N(0, I) (dùng torch.randn_like). Gradient phải chảy về mu và logvar.\"\"\"
    # TODO
    raise NotImplementedError

SIG = 0.1                                                  # độ lệch chuẩn cố định của bộ giải mã
def nll_rec(x, xr):
    \"\"\"−log p(x | z) với p(x | z) = N(xr, SIG² I) trên 3 tọa độ, đầy đủ cả hằng số. Vào (N, 3), ra (N,).\"\"\"
    # TODO
    raise NotImplementedError

class VAE(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.enc, self.dec = mlp(3, 2 * k), mlp(k, 3)
    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, 1)
        return self.dec(reparam(mu, logvar)), mu, logvar

def train_vae(beta, k=2, steps=1000, seed=0):
    torch.manual_seed(seed)
    m = VAE(k); opt = torch.optim.Adam(m.parameters(), lr=3e-3, foreach=True)
    for _ in range(steps):
        x = Xc_tr[torch.randint(0, 4000, (256,))]
        xr, mu, lv = m(x)
        loss = (nll_rec(x, xr) + beta * kl_gauss(mu, lv)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def eval_vae(m):
    torch.manual_seed(5)
    xr, mu, lv = m(Xc_te)
    per_dim = (0.5 * (lv.exp() + mu ** 2 - 1 - lv)).mean(0)
    mse = ((m.dec(mu) - Xc_te) ** 2).sum(1).mean().item()  # giải mã thẳng từ trung bình, không lấy mẫu
    return nll_rec(Xc_te, xr).mean().item(), kl_gauss(mu, lv).mean().item(), mse, per_dim""",
"""def kl_gauss(mu, logvar):
    return 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1)

def reparam(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)     # σ = exp(logvar / 2)

SIG = 0.1
def nll_rec(x, xr):
    return (0.5 * ((x - xr) / SIG) ** 2).sum(1) + 3 * (math.log(SIG) + 0.5 * math.log(2 * math.pi))

class VAE(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.enc, self.dec = mlp(3, 2 * k), mlp(k, 3)
    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, 1)
        return self.dec(reparam(mu, logvar)), mu, logvar

def train_vae(beta, k=2, steps=1000, seed=0):
    torch.manual_seed(seed)
    m = VAE(k); opt = torch.optim.Adam(m.parameters(), lr=3e-3, foreach=True)
    for _ in range(steps):
        x = Xc_tr[torch.randint(0, 4000, (256,))]
        xr, mu, lv = m(x)
        loss = (nll_rec(x, xr) + beta * kl_gauss(mu, lv)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def eval_vae(m):
    torch.manual_seed(5)
    xr, mu, lv = m(Xc_te)
    per_dim = (0.5 * (lv.exp() + mu ** 2 - 1 - lv)).mean(0)
    mse = ((m.dec(mu) - Xc_te) ** 2).sum(1).mean().item()
    return nll_rec(Xc_te, xr).mean().item(), kl_gauss(mu, lv).mean().item(), mse, per_dim""")

VAE_CHECK = """# (a) Ví dụ 10.1: KL đóng so với Monte Carlo
mu_t, sig_t = torch.tensor([[0.5, -1.0]]), torch.tensor([[0.8, 1.5]])
kl_cf = kl_gauss(mu_t, (sig_t ** 2).log()).item()
g = torch.Generator().manual_seed(0)
z = mu_t + sig_t * torch.randn(2_000_000, 2, generator=g)
log_q = (-0.5 * ((z - mu_t) / sig_t) ** 2 - sig_t.log()).sum(1)         # hằng số −½ln2π triệt tiêu
log_p = (-0.5 * z ** 2).sum(1)
kl_mc = (log_q - log_p).mean().item()
print(f"KL đóng {kl_cf:.4f} | Monte Carlo 2 triệu mẫu {kl_mc:.4f}")
assert abs(kl_cf - 0.8877) < 1e-3 and abs(kl_mc - kl_cf) / kl_cf < 0.02

# (b) Ví dụ 10.3: gradient của E[z²] theo μ, reparameterization so với score function
M = 2_000_000
mu = torch.full((M, 1), 0.5, requires_grad=True)
lv = torch.full((M, 1), math.log(0.8 ** 2))
torch.manual_seed(0)
z = reparam(mu, lv)
(z ** 2).sum().backward()
g_re = mu.grad.squeeze()                                                 # 2z cho từng mẫu
zz = z.detach().squeeze()
g_sf = zz ** 2 * (zz - 0.5) / 0.8 ** 2                                   # f(z)·(z − μ)/σ²
print(f"reparam: trung bình {g_re.mean():.4f}, độ lệch chuẩn mỗi mẫu {g_re.std():.3f}")
print(f"score function: trung bình {g_sf.mean():.4f}, độ lệch chuẩn mỗi mẫu {g_sf.std():.3f}")
assert abs(g_re.mean().item() - 1.0) < 0.02 and abs(g_sf.mean().item() - 1.0) < 0.05
assert g_sf.std() > 2 * g_re.std()

# (c) Cái bẫy torch.normal: chạy không lỗi, gradient bằng 0
m_ = torch.tensor(0.5, requires_grad=True); s_ = torch.tensor(0.8, requires_grad=True)
(torch.normal(m_.expand(1000), s_.expand(1000)) ** 2).sum().backward()
print("torch.normal: grad theo mu =", m_.grad.item(), "| grad theo σ =", s_.grad.item())
assert m_.grad.item() == 0.0

# (d) AE tuyến tính so với PCA, AE phi tuyến
e_pca, e_lin, e_nl = pca_error(2), train_ae("linear", 2), train_ae("mlp", 2)
print(f"k = 2: PCA {e_pca:.5f} | AE tuyến tính {e_lin:.5f} | AE phi tuyến {e_nl:.4f}")
assert abs(e_pca - 0.33289) < 1e-3 and abs(e_lin - e_pca) < 1e-3 and e_nl < 0.1

# (e) VAE: ba giá trị β và k = 6
res = {}
for name, beta, k in [("β=1, k=2", 1.0, 2), ("β=0, k=2", 0.0, 2), ("β=30, k=2", 30.0, 2), ("β=1, k=6", 1.0, 6)]:
    res[name] = eval_vae(train_vae(beta, k))
    r_, kl_, mse_, pd = res[name]
    print(f"{name:10s} −log p(x|z) {r_:7.3f} | KL {kl_:6.3f} | −ELBO {r_ + kl_:6.3f} | sai số tái tạo {mse_:.4f} | KL từng chiều {pd.numpy().round(3)}")
r1 = res["β=1, k=2"]
assert abs(r1[0] + r1[1] - 3.791) < 0.3 and r1[1] > 3
assert res["β=0, k=2"][1] > 10 and res["β=30, k=2"][1] < 1 and res["β=30, k=2"][2] > 0.3
pd6 = res["β=1, k=6"][3].sort().values
assert (pd6[:3] < 0.01).all() and (pd6[3:] > 1.0).all(), "k = 6: ba chiều bị tỉa (KL ≈ 0), ba chiều còn hoạt động"
print("Lab 1 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 2
GAN = ("""def d_star(x, d=2.0):
    \"\"\"Discriminator tối ưu giữa p_data = N(0, 1) và p_g = N(d, 1) (dùng scipy.stats.norm.pdf).\"\"\"
    # TODO
    raise NotImplementedError

def jsd_shift(d):
    \"\"\"Jensen–Shannon (đơn vị nat) giữa N(0, 1) và N(d, 1), tích phân số trên lưới [−12, d + 12].\"\"\"
    # TODO
    raise NotImplementedError

def gen_grad_mag(D_out, saturating):
    \"\"\"|đạo hàm của loss generator theo logit của D| khi D(G(z)) = D_out, tính bằng autograd.
    saturating=True: loss = log(1 − D). saturating=False: loss = −log D.\"\"\"
    # TODO
    raise NotImplementedError

def real(n):
    k = (torch.rand(n) < 0.5).float()
    return (k * 6 - 3) + 0.3 * torch.randn(n)                  # hỗn hợp N(−3, 0.3²) + N(3, 0.3²)

def train_gan(seed, steps=1500):
    \"\"\"G: 2→32→32→1 (ReLU); D: 1→32→32→1 (LeakyReLU 0.2, đầu ra là logit). Adam lr 2e-3, betas (0.5, 0.999) cho cả hai.
    Mỗi bước: (1) cập nhật D với batch 128 thật (nhãn 1) và 128 giả (nhãn 0, nhớ detach);
    (2) cập nhật G bằng loss non-saturating (nhãn 1 cho mẫu giả). Dùng F.binary_cross_entropy_with_logits.
    Thứ tự rút ngẫu nhiên: real(128), randn(128, 2) cho D; randn(128, 2) cho G. torch.manual_seed(seed) trước khi tạo G, D.
    Trả về (loss D ở bước cuối, 4000 mẫu của G dưới dạng tensor 1 chiều).\"\"\"
    # TODO
    raise NotImplementedError""",
"""from scipy.stats import norm

def d_star(x, d=2.0):
    p, q = norm.pdf(x, 0, 1), norm.pdf(x, d, 1)
    return p / (p + q)

def jsd_shift(d):
    xs = np.linspace(-12, d + 12, 200_001)
    p, q = norm.pdf(xs, 0, 1), norm.pdf(xs, d, 1)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * np.log(p / m), 0.0)
        kq = np.where(q > 0, q * np.log(q / m), 0.0)
    return 0.5 * (np.trapezoid(kp, xs) + np.trapezoid(kq, xs))

def gen_grad_mag(D_out, saturating):
    logit = torch.logit(torch.tensor(D_out)).requires_grad_()
    D = torch.sigmoid(logit)
    loss = torch.log(1 - D) if saturating else -torch.log(D)
    loss.backward()
    return abs(logit.grad.item())

def real(n):
    k = (torch.rand(n) < 0.5).float()
    return (k * 6 - 3) + 0.3 * torch.randn(n)

def train_gan(seed, steps=1500):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    D = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
    oG = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.5, 0.999), foreach=True)
    oD = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.5, 0.999), foreach=True)
    ones, zeros = torch.ones(128, 1), torch.zeros(128, 1)
    bce = F.binary_cross_entropy_with_logits
    for _ in range(steps):
        x = real(128).unsqueeze(1); fake = G(torch.randn(128, 2)).detach()    # detach: bước D không đụng G
        lD = bce(D(x), ones) + bce(D(fake), zeros)
        oD.zero_grad(); lD.backward(); oD.step()
        lG = bce(D(G(torch.randn(128, 2))), ones)                              # non-saturating
        oG.zero_grad(); lG.backward(); oG.step()
    with torch.no_grad():
        return lD.item(), G(torch.randn(4000, 2)).squeeze(1)""")

GAN_CHECK = """# (a) Ví dụ 10.4 và 10.5
print("D*(x), x = −1..3:", np.round(d_star(np.array([-1, 0, 1, 2, 3.0])), 4))
print("JSD khi dịch d = 0.5, 2, 5, 10:", [round(float(jsd_shift(d)), 4) for d in (0.5, 2, 5, 10)], "| ln 2 =", round(math.log(2), 4))
assert abs(d_star(0.0) - 0.8808) < 1e-4 and abs(jsd_shift(2) - 0.3368) < 1e-3 and abs(jsd_shift(10) - math.log(2)) < 1e-3
for D_out in (0.01, 0.5):
    print(f"D(G(z)) = {D_out}: |gradient| bão hòa {gen_grad_mag(D_out, True):.2f} | non-saturating {gen_grad_mag(D_out, False):.2f}")
assert abs(gen_grad_mag(0.01, True) - 0.01) < 1e-4 and abs(gen_grad_mag(0.01, False) - 0.99) < 1e-4

# (b) GAN 1D hai mode, ba seed
def coverage(f):
    return ((f + 3).abs() < 1).float().mean().item(), ((f - 3).abs() < 1).float().mean().item()

cov = {}
for seed in range(3):
    t1 = time.time()
    lD, f = train_gan(seed)
    cov[seed] = coverage(f)
    print(f"seed {seed}: gần −3 = {cov[seed][0]:.3f}, gần +3 = {cov[seed][1]:.3f}, loss D cuối = {lD:.3f}  (cân bằng lý thuyết 2ln2 = {2 * math.log(2):.3f})  [{time.time() - t1:.1f}s]")
    assert sum(cov[seed]) > 0.85, "G phải đưa hầu hết mẫu vào các mode"
print("Lab 2 ĐẠT ✔")"""

GAN_CHECK_STRICT = """# Chỉ có trong bản lời giải: số đo cụ thể (seed cố định, tái lập được)
assert min(cov[0]) < 0.05 and min(cov[1]) > 0.3 and min(cov[2]) > 0.3, "seed 0 sụp mode, seed 1 và 2 phủ cả hai mode"
print("Kiểm số đo cụ thể ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 3
DDPM_DATA = """def moons(n, seed):
    g = torch.Generator().manual_seed(seed)
    c = (torch.rand(n, generator=g) < 0.5).long(); th = torch.rand(n, generator=g) * math.pi
    x = torch.where(c == 0, torch.cos(th), 1 - torch.cos(th))
    y = torch.where(c == 0, torch.sin(th), 0.5 - torch.sin(th))
    p = torch.stack([x, y], 1) + 0.08 * torch.randn(n, 2, generator=g)
    return (p - torch.tensor([0.5, 0.25])) * 2.0, c            # lớp 0: nửa vành trên tâm (−1, −0.5), bán kính 2

X, C = moons(6000, 0)
Xte, Cte = moons(3000, 1)
T = 100                                                         # số bước khuếch tán của đồ chơi này

class Net(nn.Module):
    \"\"\"Dự đoán nhiễu ε từ (x_t, t, nhãn). Nhãn 0, 1 là hai mặt trăng, nhãn 2 là 'rỗng' ∅.\"\"\"
    def __init__(self, h=96):
        super().__init__()
        self.t_emb, self.c_emb = nn.Embedding(T, 16), nn.Embedding(3, 16)
        self.net = nn.Sequential(nn.Linear(2 + 32, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, t, c):
        return self.net(torch.cat([x, self.t_emb(t), self.c_emb(c)], 1))

def radial_error(s0):
    \"\"\"Lớp 0: sai lệch trung bình |khoảng cách tới tâm − 2| của các điểm nằm nửa trên; và tỉ lệ điểm ở nửa trên.\"\"\"
    d = s0 - torch.tensor([-1.0, -0.5]); up = d[:, 1] > 0
    return (d.norm(dim=1)[up] - 2).abs().mean().item(), up.float().mean().item()"""

DDPM = ("""def frechet(a, b):
    \"\"\"Khoảng cách Fréchet giữa hai Gauss khớp moment của hai tập điểm (tensor N×d):
    ‖m₁ − m₂‖² + Tr(C₁ + C₂ − 2(C₁C₂)^½). Dùng scipy.linalg.sqrtm và lấy phần thực.\"\"\"
    # TODO
    raise NotImplementedError

def make_schedule(T, b0, b1):
    \"\"\"β tuyến tính (torch.linspace), α = 1 − β, ᾱ = tích lũy của α. Trả về (beta, alpha, abar).\"\"\"
    # TODO
    raise NotImplementedError

def q_sample(x0, abar_t, eps):
    \"\"\"Tạo x_t từ x0 và nhiễu ε theo quá trình nhiễu hóa thuận (mục 10.3). abar_t = ᾱ_t có shape (N, 1).
    Trả về tensor cùng shape với x0.\"\"\"
    # TODO
    raise NotImplementedError

def cfg_combine(eps_c, eps_u, w):
    \"\"\"Dạng của Ho & Salimans: (1 + w)·ε(x, c) − w·ε(x, ∅).\"\"\"
    # TODO
    raise NotImplementedError

beta, alpha, abar = make_schedule(T, 1e-3, 0.1)

def train_ddpm(steps=3000, p_uncond=0.1, seed=0):
    \"\"\"Huấn luyện một Net trên (X, C) bằng Adam (lr 2e-3, batch 256, `steps` bước) rồi trả về Net đã huấn luyện.
    Net phải dùng được cho lấy mẫu có hướng dẫn (mục 10.3); p_uncond là tham số liên quan tới việc đó.
    torch.manual_seed(seed) trước khi tạo Net. Các kiểm tra bên dưới chỉ so w = 7 với w = 0 (có biên độ), không so số cụ thể và không đòi đơn điệu từng bước.\"\"\"
    # TODO
    raise NotImplementedError

@torch.no_grad()
def sample(m, n, c, w=0.0, seed=0):
    \"\"\"Sinh n điểm của lớp c với hướng dẫn w: bắt đầu từ nhiễu chuẩn, đi ngược T bước về dữ liệu (mục 10.3).
    Mọi số ngẫu nhiên lấy từ g = torch.Generator().manual_seed(seed). Trả về tensor (n, 2). Dùng cfg_combine với w.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def frechet(a, b):
    a, b = a.numpy().astype(np.float64), b.numpy().astype(np.float64)
    m1, m2 = a.mean(0), b.mean(0)
    S1, S2 = np.atleast_2d(np.cov(a.T)), np.atleast_2d(np.cov(b.T))
    return float(((m1 - m2) ** 2).sum() + np.trace(S1 + S2 - 2 * sqrtm(S1 @ S2).real))

def make_schedule(T, b0, b1):
    beta = torch.linspace(b0, b1, T); alpha = 1 - beta
    return beta, alpha, torch.cumprod(alpha, 0)

def q_sample(x0, abar_t, eps):
    return abar_t.sqrt() * x0 + (1 - abar_t).sqrt() * eps

def cfg_combine(eps_c, eps_u, w):
    return (1 + w) * eps_c - w * eps_u

beta, alpha, abar = make_schedule(T, 1e-3, 0.1)

def train_ddpm(steps=3000, p_uncond=0.1, seed=0):
    torch.manual_seed(seed); m = Net(); opt = torch.optim.Adam(m.parameters(), lr=2e-3, foreach=True)
    for _ in range(steps):
        idx = torch.randint(0, 6000, (256,)); x0 = X[idx]; c = C[idx].clone()
        t = torch.randint(0, T, (256,)); eps = torch.randn(256, 2)
        xt = q_sample(x0, abar[t].unsqueeze(1), eps)
        c[torch.rand(256) < p_uncond] = 2                       # thay nhãn bằng ∅
        loss = ((m(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def sample(m, n, c, w=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g); cc, cu = torch.full((n,), c), torch.full((n,), 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps_hat = cfg_combine(m(x, tt, cc), m(x, tt, cu) if w else 0.0, w)    # w = 0: bỏ nhánh không điều kiện
        mean = (x - beta[t] / (1 - abar[t]).sqrt() * eps_hat) / alpha[t].sqrt()
        if t > 0:
            var = (1 - abar[t - 1]) / (1 - abar[t]) * beta[t]
            x = mean + var.sqrt() * torch.randn(n, 2, generator=g)
        else:
            x = mean
    return x""")

DDPM_CHECK = """# (a) Ví dụ 10.12 (2 Gauss 2D) cho frechet; ví dụ 10.6, 10.7, 10.8 cho lịch nhiễu, q_sample, CFG
m1, m2 = np.zeros(2), np.array([1.0, 2.0])
C1, C2 = np.array([[1, 0.5], [0.5, 2]]), np.diag([2.0, 1.0])
fid_tay = float(((m1 - m2) ** 2).sum() + np.trace(C1 + C2 - 2 * sqrtm(C1 @ C2).real))
print("FID hai Gauss (công thức, ví dụ 10.12):", round(fid_tay, 4))
rg = np.random.default_rng(3)                                 # kiểm hàm frechet của BẠN trên 400 000 mẫu mỗi Gauss
fid_mau = frechet(torch.tensor(rg.multivariate_normal(m1, C1, 400_000)), torch.tensor(rg.multivariate_normal(m2, C2, 400_000)))
print("frechet() trên mẫu từ hai Gauss đó:", round(fid_mau, 4))
assert abs(fid_mau - fid_tay) < 0.1, "frechet() của bạn không cho ~5.4352 trên mẫu từ hai Gauss của ví dụ 10.12"
b1000, _, ab1000 = make_schedule(1000, 1e-4, 0.02)
x500 = q_sample(torch.tensor(2.0), ab1000[499], torch.tensor(0.5))
print(f"ᾱ₅₀₀ = {ab1000[499]:.6f} | x₅₀₀ = {x500:.4f} | ᾱ₁₀₀₀ = {ab1000[-1]:.3g} | ᾱ_T của đồ chơi T = 100: {abar[-1]:.5f}")
print("CFG, εc = 0.50, εu = 0.20, w = 0 / 1 / 3 / 7.5:", [round(cfg_combine(0.50, 0.20, w), 2) for w in (0, 1, 3, 7.5)])
assert abs(fid_tay - 5.4352) < 1e-3 and abs(x500.item() - 1.0406) < 1e-3 and abar[-1] < 0.01
assert [round(cfg_combine(0.50, 0.20, w), 2) for w in (0, 1, 3, 7.5)] == [0.5, 0.8, 1.4, 2.75]

# (b) Huấn luyện và lấy mẫu có hướng dẫn
t1 = time.time()
good = train_ddpm()
print(f"huấn luyện 3000 bước: {time.time() - t1:.1f}s")

def eval_w(m, w):
    s0, s1 = sample(m, 2000, 0, w, 1), sample(m, 2000, 1, w, 2)
    return frechet(torch.cat([s0, s1]), Xte), radial_error(s0), s0.std(0).numpy()

rows = {w: eval_w(good, w) for w in (0.0, 1.0, 3.0, 7.0)}
print("w | Fréchet(toàn bộ, so với dữ liệu thử) | sai số hướng tâm lớp 0 | std (x, y) lớp 0   [dữ liệu thật: hướng tâm 0.130, std (1.42, 0.62)]")
for w, (fr, (rad, up), sd) in rows.items():
    print(f"{w:g} | {fr:.4f} | {rad:.4f} | {sd.round(3)}")
frs = [rows[w][0] for w in rows]; rads = [rows[w][1][0] for w in rows]; sds = [rows[w][2][0] for w in rows]
assert frs[0] < 0.08, "w = 0 phải cho Fréchet nhỏ (< 0.08)"
assert frs[-1] > 3 * frs[0] and frs[-1] > 0.2, "w = 7 phải làm Fréchet lớn hơn w = 0 ít nhất 3 lần"
assert sds[-1] < 0.75 * sds[0], "độ lệch chuẩn x của lớp 0 phải co lại ít nhất 25% từ w = 0 đến w = 7"
assert rads[-1] < 0.8 * rads[0], "sai số hướng tâm phải giảm ít nhất 20% từ w = 0 đến w = 7"
print("Lab 3 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 4
SSL_DATA = """g0 = torch.Generator().manual_seed(0)
Wm, Wm2 = torch.randn(4, 24, generator=g0), torch.randn(24, 24, generator=g0) / 5
def make(n, seed):
    \"\"\"Dữ liệu 24 chiều sinh từ 4 biến ẩn z; nhãn nhị phân phụ thuộc z.\"\"\"
    gg = torch.Generator().manual_seed(seed); z = torch.randn(n, 4, generator=gg)
    h = torch.tanh(z @ Wm)
    return z, h + 0.3 * torch.tanh(h @ Wm2)

def view(x):
    return x + 0.25 * torch.randn_like(x)                      # 'góc nhìn': cùng mẫu cộng nhiễu khác nhau

ztr, xtr = make(8000, 1); zte, xte = make(2000, 2)
ytr_l, yte_l = (ztr[:, 0] + 0.5 * ztr[:, 1] > 0).long(), (zte[:, 0] + 0.5 * zte[:, 1] > 0).long()
def make_enc():
    return nn.Sequential(nn.Linear(24, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 16))"""

SSL = ("""def infonce(a, b, tau):
    \"\"\"InfoNCE một chiều (a → b), không đối xứng: a[i] và b[i] là cặp dương, mọi b[j] (j ≠ i) là âm của a[i].
    Chuẩn hóa L2, logits = a·bᵀ / τ, cross-entropy với nhãn là đường chéo. Trả về (loss, logits).\"\"\"
    # TODO
    raise NotImplementedError

def bert_mask(x, vocab, mask_id, p=0.15, gen=None):
    \"\"\"Quy tắc 80/10/10 của BERT. Chọn mỗi vị trí với xác suất p; trong số chọn: 80% → mask_id,
    10% → token ngẫu nhiên trong [0, vocab), 10% giữ nguyên. Trả về (đầu vào đã che, nhãn với −100 ở vị trí không chọn).
    Thứ tự rút: sel = rand(x.shape) < p; r = rand(x.shape); rồi randint cho các vị trí 'ngẫu nhiên'.\"\"\"
    # TODO
    raise NotImplementedError

def linear_probe(ftr, ytr, fte, yte, steps=300):
    \"\"\"Chuẩn hóa đặc trưng bằng trung bình/độ lệch chuẩn của train, huấn luyện một nn.Linear(d, 2) bằng Adam lr 0.05
    (toàn bộ train mỗi bước, torch.manual_seed(0)), trả về độ chính xác trên test.\"\"\"
    # TODO
    raise NotImplementedError

def retrieval_at1(enc):
    with torch.no_grad():
        a, b = F.normalize(enc(view(xte)), dim=1), F.normalize(enc(view(xte)), dim=1)
        return ((a @ b.T).argmax(1) == torch.arange(len(a))).float().mean().item()

def train_contrastive(tau, N=128, steps=600, seed=0):
    torch.manual_seed(seed); enc = make_enc(); opt = torch.optim.Adam(enc.parameters(), lr=2e-3, foreach=True)
    first = None
    for s in range(steps):
        x = xtr[torch.randint(0, 8000, (N,))]
        loss, _ = infonce(enc(view(x)), enc(view(x)), tau)
        first = loss.item() if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
    return enc, first

# Palindrome: nửa sau là đảo của nửa đầu. Cùng một Transformer, hai mục tiêu khác nhau.
V, L, MASK = 10, 8, 10
def pal_batch(n, gen=None):
    h = torch.randint(0, V, (n, L // 2), generator=gen)
    return torch.cat([h, h.flip(1)], 1)

class TinyTF(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V + 1, d), nn.Embedding(L, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True), 2,
                                    enable_nested_tensor=False)
        self.out = nn.Linear(d, V)
    def forward(self, x, causal):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]) if causal else None
        return self.out(self.enc(h, mask=mask, is_causal=causal))

def causal_lm_loss(m, x):
    \"\"\"Dự đoán token i+1 từ các token ≤ i (mặt nạ nhân quả). Cross-entropy trung bình trên mọi vị trí.\"\"\"
    # TODO
    raise NotImplementedError

def masked_lm_loss(m, x, p=0.25):
    \"\"\"Che mỗi vị trí với xác suất p (thay bằng MASK; mỗi dòng che ít nhất vị trí 0 nếu chưa che gì),
    mạng nhìn cả hai phía (causal=False); loss chỉ tính trên vị trí bị che (ignore_index = −100).\"\"\"
    # TODO
    raise NotImplementedError""",
"""def infonce(a, b, tau):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.T / tau                                        # chia cho τ, không nhân
    return F.cross_entropy(logits, torch.arange(len(a))), logits

def bert_mask(x, vocab, mask_id, p=0.15, gen=None):
    sel = torch.rand(x.shape, generator=gen) < p
    r = torch.rand(x.shape, generator=gen)
    inp = x.clone()
    inp[sel & (r < 0.8)] = mask_id
    rnd = sel & (r >= 0.8) & (r < 0.9)
    inp[rnd] = torch.randint(0, vocab, (int(rnd.sum()),), generator=gen)
    labels = torch.where(sel, x, torch.full_like(x, -100))
    return inp, labels

def linear_probe(ftr, ytr, fte, yte, steps=300):
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    a, b = (ftr - mu) / sd, (fte - mu) / sd
    torch.manual_seed(0)
    lin = nn.Linear(a.shape[1], 2); opt = torch.optim.Adam(lin.parameters(), lr=0.05, foreach=True)
    for _ in range(steps):
        loss = F.cross_entropy(lin(a), ytr); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (lin(b).argmax(1) == yte).float().mean().item()

def retrieval_at1(enc):
    with torch.no_grad():
        a, b = F.normalize(enc(view(xte)), dim=1), F.normalize(enc(view(xte)), dim=1)
        return ((a @ b.T).argmax(1) == torch.arange(len(a))).float().mean().item()

def train_contrastive(tau, N=128, steps=600, seed=0):
    torch.manual_seed(seed); enc = make_enc(); opt = torch.optim.Adam(enc.parameters(), lr=2e-3, foreach=True)
    first = None
    for s in range(steps):
        x = xtr[torch.randint(0, 8000, (N,))]
        loss, _ = infonce(enc(view(x)), enc(view(x)), tau)
        first = loss.item() if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
    return enc, first

V, L, MASK = 10, 8, 10
def pal_batch(n, gen=None):
    h = torch.randint(0, V, (n, L // 2), generator=gen)
    return torch.cat([h, h.flip(1)], 1)

class TinyTF(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V + 1, d), nn.Embedding(L, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True), 2,
                                    enable_nested_tensor=False)
        self.out = nn.Linear(d, V)
    def forward(self, x, causal):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]) if causal else None
        return self.out(self.enc(h, mask=mask, is_causal=causal))

def causal_lm_loss(m, x):
    logits = m(x[:, :-1], True)
    return F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))

def masked_lm_loss(m, x, p=0.25):
    msk = torch.rand(x.shape) < p
    msk[:, 0] |= (msk.sum(1) == 0)
    logits = m(x.masked_fill(msk, MASK), False)
    return F.cross_entropy(logits.reshape(-1, V), x.masked_fill(~msk, -100).reshape(-1), ignore_index=-100)""")

SSL_CHECK = """# (a) Ví dụ 10.9: ma trận tương đồng 4×4 (coi S là ma trận cos sẵn có nên đưa thẳng vào cross-entropy, không qua `infonce`)
S = torch.tensor([[0.9, 0.1, 0.0, 0.2], [0.2, 0.8, 0.3, 0.1], [0.0, 0.2, 0.7, 0.4], [0.1, 0.1, 0.3, 0.6]])
for tau in (1.0, 0.5, 0.1):
    print(f"τ = {tau}: loss trung bình {F.cross_entropy(S / tau, torch.arange(4)).item():.4f}")
assert abs(F.cross_entropy(S / 0.5, torch.arange(4)).item() - 0.6816) < 1e-4
ones = torch.ones(128, 16)
eye = torch.eye(128)
c_col = infonce(ones, ones, 0.5)[0].item()
print(f"embedding sụp đổ: loss {c_col:.4f} (ln 128 = {math.log(128):.4f}) | trực giao hoàn hảo: τ = 0.1 → {infonce(eye, eye, 0.1)[0].item():.4f}, τ = 0.5 → {infonce(eye, eye, 0.5)[0].item():.4f}")
assert abs(c_col - math.log(128)) < 1e-4 and abs(infonce(eye, eye, 0.5)[0].item() - 2.9007) < 1e-3

# (b) Quét τ: retrieval và linear probe
out = {}
for tau in (0.05, 0.5):
    t1 = time.time()
    enc, first = train_contrastive(tau)
    with torch.no_grad():
        ftr, fte = enc(xtr), enc(xte)
    out[tau] = (first, retrieval_at1(enc), linear_probe(ftr, ytr_l, fte, yte_l))
    print(f"τ = {tau}: loss bước đầu {first:.3f} | retrieval@1 {out[tau][1]:.3f} | probe {out[tau][2]:.3f}  [{time.time() - t1:.1f}s]")
torch.manual_seed(0); enc0 = make_enc()
with torch.no_grad():
    f0tr, f0te = enc0(xtr), enc0(xte)
p_rand = linear_probe(f0tr, ytr_l, f0te, yte_l)
p_raw = linear_probe(xtr, ytr_l, xte, yte_l)
print(f"encoder ngẫu nhiên chưa huấn luyện: probe {p_rand:.3f} | đầu vào thô: probe {p_raw:.3f}")
assert out[0.05][1] > out[0.5][1] and out[0.05][2] < out[0.5][2]

# (c) Mặt nạ BERT trên 200 000 token
gen = torch.Generator().manual_seed(0)
xx = torch.randint(0, 1000, (200_000,), generator=gen)
inp, lab = bert_mask(xx, 1000, 1000, gen=gen)
sel = lab != -100
f_mask, f_same = (inp[sel] == 1000).float().mean().item(), (inp[sel] == xx[sel]).float().mean().item()
print(f"đã chọn {sel.float().mean():.4f} | trong số chọn: [MASK] {f_mask:.3f}, giữ nguyên {f_same:.3f}, ngẫu nhiên {1 - f_mask - f_same:.3f}")
assert abs(sel.float().mean().item() - 0.15) < 0.003 and abs(f_mask - 0.8) < 0.01 and abs(f_same - 0.1) < 0.012

# (d) Palindrome: causal LM và masked LM
def fit(loss_fn, steps=400, seed=0):
    torch.manual_seed(seed); m = TinyTF(); opt = torch.optim.Adam(m.parameters(), lr=2e-3, foreach=True)
    for _ in range(steps):
        loss = loss_fn(m, pal_batch(64)); opt.zero_grad(); loss.backward(); opt.step()
    return m

t1 = time.time()
m_c, m_m = fit(causal_lm_loss), fit(masked_lm_loss)
with torch.no_grad():
    xe = pal_batch(4000, torch.Generator().manual_seed(9))
    lc = F.cross_entropy(m_c(xe[:, :-1], True).reshape(-1, V), xe[:, 1:].reshape(-1), reduction="none").view(4000, -1)
    pos = torch.randint(0, L, (4000,)); xin = xe.clone(); xin[torch.arange(4000), pos] = MASK
    lm = F.cross_entropy(m_m(xin, False)[torch.arange(4000), pos], xe[torch.arange(4000), pos]).item()
floor = (L // 2 - 1) / (L - 1) * math.log(V)
print(f"causal LM: loss {lc.mean():.4f} (nửa đầu {lc[:, :L // 2 - 1].mean():.3f}, nửa sau {lc[:, L // 2 - 1:].mean():.4f}); sàn lý thuyết (3/7)·ln 10 = {floor:.4f}")
print(f"masked LM (che đúng một vị trí): loss {lm:.4f}   [{time.time() - t1:.1f}s]")
assert lc.mean().item() > floor - 0.01 and lm < 0.05
print("Lab 4 ĐẠT ✔")"""

# ---------------------------------------------------------------- Lab 5
EVAL = ("""def perplexity(logits, y):
    \"\"\"exp(cross-entropy trung bình). logits: (N, V), y: (N,).\"\"\"
    # TODO
    raise NotImplementedError

def bits_per_dim(nll_nats, dims):
    \"\"\"NLL (nat) của một mẫu có `dims` chiều → bits/dim.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def perplexity(logits, y):
    return math.exp(F.cross_entropy(logits, y).item())

def bits_per_dim(nll_nats, dims):
    return nll_nats / dims / math.log(2)""")

EVAL_CHECK = """rng = np.random.default_rng(0)
def T_(a): return torch.tensor(a, dtype=torch.float64)

# (a) Thiên lệch theo cỡ mẫu: hai tập cùng một N(0, I) trong d = 64
bias = {}
for N in (100, 500, 2000, 10000):
    vals = [frechet(T_(rng.standard_normal((N, 64))), T_(rng.standard_normal((N, 64)))) for _ in range(3)]
    bias[N] = float(np.mean(vals))
    print(f"N = {N:5d}: FID giữa hai mẫu CÙNG phân phối = {bias[N]:.3f}  | N·FID = {N * bias[N]:.0f}")
assert 2000 < 500 * bias[500] < 2400 and 2000 < 2000 * bias[2000] < 2400

# (a') N·FID theo số chiều d (N = 2000) so với xấp xỉ bậc hai d(d+1)/2 + 2d của mục 10.5.
# Nhiều lần lặp cho d nhỏ vì N·FID dao động mạnh giữa các lần chạy (d = 4 chỉ 3 lần là lệch hẳn).
nf, rng2 = {}, np.random.default_rng(1)       # rng riêng để không đổi số của (b), (c)
for d, reps in ((4, 200), (16, 40), (32, 20), (64, 10)):
    nf[d] = 2000 * float(np.mean([frechet(T_(rng2.standard_normal((2000, d))), T_(rng2.standard_normal((2000, d)))) for _ in range(reps)]))
for d, v in nf.items():
    print(f"d = {d:2d}: N·FID = {v:7.1f} | d(d+1)/2 + 2d = {d * (d + 1) / 2 + 2 * d:7.1f} | d²/2 + 2d = {d * d / 2 + 2 * d:7.1f}")
assert all(abs(v / (d * (d + 1) / 2 + 2 * d) - 1) < 0.08 for d, v in nf.items())

# (b) Mù hình dạng: Gauss chuẩn so với vòng tròn bán kính √2 (cùng trung bình và hiệp phương sai)
a = T_(rng.standard_normal((20000, 2)))
th = rng.uniform(0, 2 * np.pi, 20000)
b = T_(np.sqrt(2) * np.stack([np.cos(th), np.sin(th)], 1))
f_circle = frechet(a, b)
print(f"Gauss so với vòng tròn: FID = {f_circle:.5f}")

# (c) Bỏ một mode, và Gauss khớp moment
data = np.concatenate([rng.normal(-3, 0.3, (5000, 1)), rng.normal(3, 0.3, (5000, 1))])
drop = frechet(T_(data), T_(rng.normal(3, 0.3, (10000, 1))))
gfit = frechet(T_(data), T_(rng.normal(0, np.sqrt(data.var()), (10000, 1))))
print(f"bỏ mode trái: FID = {drop:.2f} | một Gauss khớp moment: FID = {gfit:.4f}")
assert f_circle < 0.01 and abs(drop - 16.35) < 0.5 and gfit < 0.05

# (d) Perplexity và bits/dim (ví dụ 10.11)
V50 = 50257
ppl = perplexity(torch.zeros(5, V50), torch.randint(0, V50, (5,)))
print(f"perplexity của mô hình đoán đều: {ppl:.1f} | bits/dim của 6400 nat trên 32×32×3: {bits_per_dim(6400, 3072):.4f}")
print("2.9 nat/token, 4.2 byte/token → bit/byte:", round(2.9 / math.log(2) / 4.2, 3))
assert abs(ppl - V50) < 1 and abs(bits_per_dim(6400, 3072) - 3.0056) < 1e-3
print("Lab 5 ĐẠT ✔")"""

# ---------------------------------------------------------------- Săn bug
BUG_CODE = """def train_buggy(steps=3000, seed=0):
    torch.manual_seed(seed); m = Net(); opt = torch.optim.Adam(m.parameters(), lr=2e-3, foreach=True)
    losses = []
    for i in range(steps):
        idx = torch.randint(0, 6000, (256,)); x0 = X[idx]; c = C[idx]
        t = torch.randint(0, T, (256,)); eps = torch.randn(1, 2)
        a = abar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a) * eps
        loss = ((m(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return m, losses

@torch.no_grad()
def sample_buggy(m, n, c, w=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g); cc, cu = torch.full((n,), c), torch.full((n,), 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps_hat = (1 + w) * m(x, tt, cc) - w * m(x, tt, cu)
        mean = x - beta[t] / (1 - abar[t]).sqrt() * eps_hat
        if t > 0:
            var = (1 - abar[t - 1]) / (1 - abar[t]) * beta[t]
            x = mean + var.sqrt() * torch.randn(n, 2, generator=g)
        else:
            x = mean
    return x

def eval_pair(m, sampler, w):
    s0, s1 = sampler(m, 2000, 0, w, 1), sampler(m, 2000, 1, w, 2)
    return frechet(torch.cat([s0, s1]), Xte), radial_error(s0)[0], s0.std(0).numpy()

t1 = time.time()
mb, loss_b = train_buggy()
print(f"[{time.time() - t1:.1f}s] loss huấn luyện: bước 100 = {np.mean(loss_b[:100]):.3f}, 100 bước cuối = {np.mean(loss_b[-100:]):.3f}")
res_buggy = {}
for w in (0.0, 3.0):
    res_buggy[w] = eval_pair(mb, sample_buggy, w)
    fr, rad, sd = res_buggy[w]
    print(f"bản lỗi, w = {w:g}: Fréchet {fr:.4f} | sai số hướng tâm lớp 0 {rad:.4f} | std lớp 0 {sd.round(3)}")"""

BUGFIX = ("""def train_fixed(steps=3000, seed=0):
    \"\"\"Sửa MỌI lỗi của train_buggy. Trả về (model, danh sách loss).\"\"\"
    # TODO
    raise NotImplementedError

@torch.no_grad()
def sample_fixed(m, n, c, w=0.0, seed=0):
    \"\"\"Sửa MỌI lỗi của sample_buggy.\"\"\"
    # TODO
    raise NotImplementedError""",
"""def train_fixed(steps=3000, seed=0):
    torch.manual_seed(seed); m = Net(); opt = torch.optim.Adam(m.parameters(), lr=2e-3, foreach=True)
    losses = []
    for i in range(steps):
        idx = torch.randint(0, 6000, (256,)); x0 = X[idx]; c = C[idx].clone()
        t = torch.randint(0, T, (256,)); eps = torch.randn(256, 2)           # lỗi 3: mỗi mẫu một ε riêng
        a = abar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps                           # lỗi 1: √(1 − ᾱ), không phải (1 − ᾱ)
        c[torch.rand(256) < 0.1] = 2                                        # lỗi 2: thỉnh thoảng thay nhãn bằng ∅
        loss = ((m(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return m, losses

@torch.no_grad()
def sample_fixed(m, n, c, w=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g); cc, cu = torch.full((n,), c), torch.full((n,), 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps_hat = (1 + w) * m(x, tt, cc) - w * (m(x, tt, cu) if w else 0.0)
        mean = (x - beta[t] / (1 - abar[t]).sqrt() * eps_hat) / alpha[t].sqrt()   # lỗi 4: chia cho √αₜ
        if t > 0:
            var = (1 - abar[t - 1]) / (1 - abar[t]) * beta[t]
            x = mean + var.sqrt() * torch.randn(n, 2, generator=g)
        else:
            x = mean
    return x""")

BUG_CHECK = """mf, loss_f = train_fixed()
r0 = eval_pair(mf, sample_fixed, 0.0)
r3 = eval_pair(mf, sample_fixed, 3.0)
print(f"loss huấn luyện bản sửa: 100 bước cuối = {np.mean(loss_f[-100:]):.3f} (bản lỗi: {np.mean(loss_b[-100:]):.3f})")
print(f"bản sửa, w = 0: Fréchet {r0[0]:.4f} | hướng tâm {r0[1]:.4f} | std {r0[2].round(3)}")
print(f"bản sửa, w = 3: Fréchet {r3[0]:.4f} | hướng tâm {r3[1]:.4f} | std {r3[2].round(3)}")
assert r0[0] < 0.08 and r0[1] < 0.15
assert r3[0] < 0.5 and r3[1] < 0.09
print("Săn bug ĐẠT ✔")"""

ABLATION = """# Tách từng lỗi: chỉ bật MỘT lỗi trên bản đúng (cùng seed) để thấy lỗi nào gây triệu chứng nào.
def train_flag(nosqrt=False, pdrop=0.1, shared=False, steps=3000, seed=0):
    torch.manual_seed(seed); m = Net(); opt = torch.optim.Adam(m.parameters(), lr=2e-3, foreach=True)
    for i in range(steps):
        idx = torch.randint(0, 6000, (256,)); x0 = X[idx]; c = C[idx].clone()
        t = torch.randint(0, T, (256,)); eps = torch.randn(1, 2) if shared else torch.randn(256, 2)
        a = abar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + ((1 - a) if nosqrt else (1 - a).sqrt()) * eps
        if pdrop > 0: c[torch.rand(256) < pdrop] = 2
        loss = ((m(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def sample_nodiv(m, n, c, w=0.0, seed=0):          # sample_fixed nhưng không chia cho √αₜ
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g); cc, cu = torch.full((n,), c), torch.full((n,), 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps_hat = cfg_combine(m(x, tt, cc), m(x, tt, cu) if w else 0.0, w)
        mean = x - beta[t] / (1 - abar[t]).sqrt() * eps_hat
        if t > 0:
            x = mean + ((1 - abar[t - 1]) / (1 - abar[t]) * beta[t]).sqrt() * torch.randn(n, 2, generator=g)
        else:
            x = mean
    return x

tab = {}
tab["bản đúng, w = 0"], tab["bản đúng, w = 3"] = r0, r3
m_ns = train_flag(nosqrt=True);           tab["chỉ lỗi 1 (thiếu căn), w = 0"] = eval_pair(m_ns, sample_fixed, 0.0)
m_sh = train_flag(shared=True)
tab["chỉ lỗi 3 (ε chung batch), w = 0"] = eval_pair(m_sh, sample_fixed, 0.0)
tab["chỉ lỗi 3 (ε chung batch), w = 3"] = eval_pair(m_sh, sample_fixed, 3.0)
m_nd = train_flag(pdrop=0.0);             tab["chỉ lỗi 2 (không dùng ∅), w = 0"] = eval_pair(m_nd, sample_fixed, 0.0)
tab["chỉ lỗi 2 (không dùng ∅), w = 3"] = eval_pair(m_nd, sample_fixed, 3.0)
tab["chỉ lỗi 4 (thiếu √α), w = 0"] = eval_pair(mf, sample_nodiv, 0.0)
tab["cả bốn lỗi, w = 0"], tab["cả bốn lỗi, w = 3"] = res_buggy[0.0], res_buggy[3.0]
print(f"{'cấu hình':36s} | Fréchet | hướng tâm | std lớp 0")
for k, (fr, rad, sd) in tab.items():
    print(f"{k:36s} | {fr:7.4f} | {rad:9.4f} | {sd.round(3)}")
assert tab["chỉ lỗi 1 (thiếu căn), w = 0"][0] < tab["bản đúng, w = 0"][0], "lỗi 1: Fréchet nhỏ hơn bản đúng"
assert tab["chỉ lỗi 1 (thiếu căn), w = 0"][1] > 1.5 * tab["bản đúng, w = 0"][1], "…nhưng hình sai hơn"
assert tab["chỉ lỗi 2 (không dùng ∅), w = 0"][0] < 0.08 and tab["chỉ lỗi 2 (không dùng ∅), w = 3"][0] > 2
assert tab["chỉ lỗi 4 (thiếu √α), w = 0"][0] > 3 * tab["bản đúng, w = 0"][0]
assert tab["chỉ lỗi 3 (ε chung batch), w = 0"][0] > 5 * tab["bản đúng, w = 0"][0], "lỗi 3 không vô hại"
assert tab["cả bốn lỗi, w = 0"][0] > 5 * tab["bản đúng, w = 0"][0]
print(f"Tổng thời gian notebook: {time.time() - T0:.1f} s")
print("Phân tích lỗi ĐẠT ✔")"""


def build(solution: bool):
    pick = (lambda pair: pair[1]) if solution else (lambda pair: pair[0])
    nb = nbf.v4.new_notebook()
    head = "Lời giải" if solution else "Bài lab"
    cells = [
        md(f"# Chương 10 — {head}: Mô hình sinh và học tự giám sát\n\n"
           + ("Bản lời giải đã chạy kiểm (PyTorch CPU, 1 luồng, dưới 2 phút). Chỉ mở sau khi đã tự làm `lab.ipynb`."
              if solution else
              "Viết code vào các ô có `# TODO`, rồi chạy ô kiểm tra ngay bên dưới. "
              "Ô kiểm tra in **ĐẠT ✔** là xong. Bí quá 20 phút mới mở `loigiai.ipynb`. "
              "Chỉ cần PyTorch bản CPU và scipy; toàn bộ notebook chạy dưới 2 phút khi đã làm xong.")),
        code(SETUP),
        md("## Lab 1 — Autoencoder, VAE, reparameterization\n"
           "Dữ liệu: cuộn bánh 3D (một mặt phẳng hai chiều bị cuộn lại, nhiễu 0.02). Hàm có sẵn: `mlp`, `VAE`, `train_vae`, `eval_vae`. "
           "Bạn viết PCA, AE, KL đóng, `reparam` và hàm tái tạo. Ô kiểm tra đối chiếu với ví dụ 10.1 và 10.3."),
        code(AE_DATA), code(pick(AE)), code(pick(VAE_CODE)), code(VAE_CHECK),
        md("## Lab 2 — GAN một chiều, hai mode\n"
           "Discriminator tối ưu, Jensen–Shannon bão hòa, gradient của hai dạng loss generator, rồi huấn luyện GAN trên "
           "hỗn hợp hai Gauss. Mục tiêu: nhìn tận mắt rằng loss không cho biết mode nào đã bị bỏ."),
        code(pick(GAN)), code(GAN_CHECK),
    ]
    if solution:
        cells.append(code(GAN_CHECK_STRICT))
    cells += [
        md("## Lab 3 — Fréchet và DDPM có classifier-free guidance\n"
           "Viết `frechet`, lịch nhiễu, `q_sample`, vòng huấn luyện và bộ lấy mẫu có hướng dẫn. "
           "Dữ liệu: hai mặt trăng có nhãn (lớp 0 là mặt trăng trên), T = 100."),
        code(DDPM_DATA), code(pick(DDPM)), code(DDPM_CHECK),
        md("## Lab 4 — InfoNCE, mặt nạ BERT, causal so với masked\n"
           "Học tương phản trên dữ liệu 24 chiều có hai góc nhìn nhiễu; mặt nạ 80/10/10; "
           "và palindrome để thấy cùng một Transformer học được những thứ khác nhau tùy mặt nạ. "
           "Linear probe viết bằng PyTorch (không dùng scikit-learn)."),
        code(SSL_DATA), code(pick(SSL)), code(SSL_CHECK),
        md("## Lab 5 — Đo đạc: thiên lệch của FID, perplexity, bits/dim\n"
           "Dùng lại `frechet` của Lab 3 để thấy ba giới hạn của nó; viết `perplexity` và `bits_per_dim`."),
        code(pick(EVAL)), code(EVAL_CHECK),
        md("## Săn bug\n`train_buggy` và `sample_buggy` chạy không báo lỗi, loss huấn luyện giảm bình thường. "
           "Triệu chứng: ô bên dưới in ra Fréchet và sai số hướng tâm của lớp 0 với w = 0 và w = 3. "
           "Tìm cả **bốn** lỗi, viết `train_fixed` và `sample_fixed`."),
        code(BUG_CODE), code(pick(BUGFIX)), code(BUG_CHECK),
    ]
    if solution:
        cells += [md("## Phân tích từng lỗi\nBật riêng từng lỗi trên bản đúng (cùng seed) để thấy lỗi nào gây triệu chứng nào."),
                  code(ABLATION)]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    name = "loigiai.ipynb" if solution else "lab.ipynb"
    nbf.write(nb, OUT / name)
    return OUT / name


if __name__ == "__main__":
    print(build(False), build(True))
