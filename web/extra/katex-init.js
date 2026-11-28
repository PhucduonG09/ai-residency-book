// Vẽ công thức bằng KaTeX; chạy lại mỗi lần chuyển trang (navigation.instant).
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
  });
});
