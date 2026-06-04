#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FOD (Foreign Object Debris) 图像标注工具
=========================================
功能：
  1. 浏览本地图片，左右按钮快速切换
  2. 支持放大/缩小，便于标注细小物体
  3. 12个FOD类别按钮，点击选择类别后画框标注
  4. 标注框(BBox)微调：四个参数独立调节
  5. "下一个"存储标注，"确定"保存到同名txt（JSON结构体，{}分级）
  6. 自动加载已有标注文件
  7. 切换图片自动保存

依赖: tkinter (标准库), Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# ──────────────────────────────────────────────────────────────
# 配置: 12个FOD类别（0-11编号对应存储）
# ──────────────────────────────────────────────────────────────
DEFAULT_CATEGORIES = [
    "金属", "橡胶", "塑料", "木头",
    "纸板", "织物", "动物", "玻璃",
    "石头", "泡沫", "油渍", "水",
]

# 类别 ↔ 编号映射（txt中存储0-11的整数编号）
CATEGORY_TO_ID = {name: i for i, name in enumerate(DEFAULT_CATEGORIES)}
ID_TO_CATEGORY = {i: name for i, name in enumerate(DEFAULT_CATEGORIES)}

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


@dataclass
class Annotation:
    """单个标注数据结构 — 内部使用像素坐标，序列化为JSON结构体"""
    category: str          # 类别名称（显示用）
    x1: int                # 左上X (像素)
    y1: int                # 左上Y (像素)
    x2: int                # 右下X (像素)
    y2: int                # 右下Y (像素)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为结构化字典（含{}分级嵌套）"""
        return {
            "id": CATEGORY_TO_ID.get(self.category, -1),
            "name": self.category,
            "bbox": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            }
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Optional['Annotation']:
        """从结构化字典反序列化"""
        try:
            class_id = int(d["id"])
            bbox = d["bbox"]
            category = ID_TO_CATEGORY.get(class_id, d.get("name", f"未知{class_id}"))
            return Annotation(
                category=category,
                x1=int(bbox["x1"]), y1=int(bbox["y1"]),
                x2=int(bbox["x2"]), y2=int(bbox["y2"]),
            )
        except (KeyError, ValueError, TypeError):
            return None


class FODAnnotationTool:
    """FOD标注工具主界面"""

    # ── 初始化 ──────────────────────────────────────────────
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FOD 图像标注工具")
        self.root.geometry("2080x1300")
        self.root.minsize(1690, 1040)

        # ── 状态变量 ──
        self.image_folder: str = ""
        self.image_list: List[str] = []
        self.current_image_idx: int = -1
        self.pil_image: Optional[Image.Image] = None   # 原始PIL图像
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.zoom_scale: float = 1.0
        self.canvas_img_id: Optional[int] = None       # Canvas上图像对象的ID

        self.annotations: List[Annotation] = []        # 当前图像的所有标注
        self.current_category: Optional[str] = None
        self.current_bbox: Optional[List[int]] = None  # [x1, y1, x2, y2] 图像坐标
        self.preview_rect_id: Optional[int] = None     # 红色预览框的Canvas ID
        self.saved_rect_ids: List[int] = []            # 已保存标注的Canvas框ID列表

        self.drawing: bool = False
        self.drag_start: Tuple[float, float] = (0, 0)

        self.save_directory: str = ""                  # 自定义保存目录（空=图片同目录）

        # ── 构建界面 ──
        self._setup_menu()
        self._setup_ui()
        self._bind_events()

        # 状态栏
        self._update_status("就绪 — 请通过「文件 → 打开文件夹」选择图片目录")

    # ══════════════════════════════════════════════════════════
    # 菜单
    # ══════════════════════════════════════════════════════════
    def _setup_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开文件夹...", command=self.open_folder, accelerator="Ctrl+O")
        file_menu.add_command(label="选择保存目录...", command=self.select_save_dir)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)
        self.root.bind('<Control-o>', lambda e: self.open_folder())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════
    # 主界面布局
    # ══════════════════════════════════════════════════════════
    def _setup_ui(self):
        # 主容器：左右分栏
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # ── 左侧: 图像显示区 ──
        left_frame = ttk.Frame(main_pw)
        main_pw.add(left_frame, weight=3)

        # Canvas + 滚动条
        canvas_container = ttk.Frame(left_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_container, bg='#2d2d2d',
                                cursor='crosshair', highlightthickness=0)
        self.canvas_hbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL,
                                         command=self.canvas.xview)
        self.canvas_vbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL,
                                         command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.canvas_hbar.set,
                              yscrollcommand=self.canvas_vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas_hbar.grid(row=1, column=0, sticky='ew')
        self.canvas_vbar.grid(row=0, column=1, sticky='ns')

        # 缩放控制栏
        ctrl_bar = ttk.Frame(left_frame)
        ctrl_bar.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(ctrl_bar, text="放大 +", command=self.zoom_in, width=7).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_bar, text="缩小 -", command=self.zoom_out, width=7).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_bar, text="适应窗口", command=self.fit_to_window, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Button(ctrl_bar, text="原始大小", command=self.zoom_100, width=8).pack(side=tk.LEFT, padx=1)
        self.zoom_label = ttk.Label(ctrl_bar, text="100%", width=8, anchor='center')
        self.zoom_label.pack(side=tk.LEFT, padx=10)

        # 导航栏
        nav_bar = ttk.Frame(left_frame)
        nav_bar.pack(fill=tk.X, padx=4, pady=4)
        self.btn_prev = ttk.Button(nav_bar, text="◀ 上一张", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.image_label = ttk.Label(nav_bar, text="未加载图片", anchor='center')
        self.image_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.btn_next = ttk.Button(nav_bar, text="下一张 ▶", command=self.next_image)
        self.btn_next.pack(side=tk.RIGHT, padx=2)

        # ── 右侧: 控制面板 ──
        right_frame = ttk.Frame(main_pw, width=380)
        main_pw.add(right_frame, weight=1)

        # 类别选择区域（可滚动）
        cat_lf = ttk.LabelFrame(right_frame, text="标注类别 (点击选择)")
        cat_lf.pack(fill=tk.BOTH, expand=True, padx=3, pady=2)

        # 用 Canvas 实现滚动 (12个类别，3行×4列)
        cat_canvas = tk.Canvas(cat_lf, height=230, highlightthickness=0)
        cat_scroll = ttk.Scrollbar(cat_lf, orient=tk.VERTICAL, command=cat_canvas.yview)
        self.cat_inner = ttk.Frame(cat_canvas)
        self.cat_inner.bind('<Configure>',
                            lambda e: cat_canvas.configure(scrollregion=cat_canvas.bbox('all')))
        self._cat_win = cat_canvas.create_window((0, 0), window=self.cat_inner, anchor='nw')

        # 让内部frame宽度跟随canvas
        def _on_cat_canvas_conf(e):
            cat_canvas.itemconfig(self._cat_win, width=e.width)
        cat_canvas.bind('<Configure>', _on_cat_canvas_conf)

        cat_canvas.configure(yscrollcommand=cat_scroll.set)
        cat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cat_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定鼠标滚轮
        def _cat_mousewheel(e):
            cat_canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        cat_canvas.bind('<Enter>',
                        lambda e: cat_canvas.bind_all('<MouseWheel>', _cat_mousewheel))
        cat_canvas.bind('<Leave>',
                        lambda e: cat_canvas.unbind_all('<MouseWheel>'))

        # 生成12个类别按钮 (4列布局)
        self.cat_buttons: List[ttk.Button] = []
        self._cat_btn_ref: dict = {}  # category -> button
        for i, cat_name in enumerate(DEFAULT_CATEGORIES):
            row, col = divmod(i, 4)
            btn = ttk.Button(self.cat_inner, text=cat_name, width=8,
                             command=lambda c=cat_name: self.select_category(c))
            btn.grid(row=row, column=col, padx=3, pady=3, sticky='ew')
            self.cat_buttons.append(btn)
            self._cat_btn_ref[cat_name] = btn
        # 均匀列宽
        for c in range(4):
            self.cat_inner.grid_columnconfigure(c, weight=1)

        # 当前选中类别提示
        self.cat_hint = ttk.Label(right_frame, text="当前类别: 未选择",
                                  foreground='#888', font=('Microsoft YaHei UI', 22, 'bold'))
        self.cat_hint.pack(fill=tk.X, padx=5, pady=(2, 0))

        # ── 标注框微调 ──
        adj_lf = ttk.LabelFrame(right_frame, text="标注框微调 (图像坐标)")
        adj_lf.pack(fill=tk.X, padx=3, pady=4)

        self.adjust_vars: dict = {}
        self.adjust_entries: dict = {}
        params = [
            ('x1', '左'), ('y1', '上'),
            ('x2', '右'), ('y2', '下'),
        ]
        for key, label in params:
            row_f = ttk.Frame(adj_lf)
            row_f.pack(fill=tk.X, padx=4, pady=2)
            ttk.Label(row_f, text=f"{label}:", width=7).pack(side=tk.LEFT)

            var = tk.IntVar(value=0)
            self.adjust_vars[key] = var

            ttk.Button(row_f, text="◀", width=3,
                       command=lambda k=key: self.adjust_box(k, -1)).pack(side=tk.LEFT, padx=1)
            entry = ttk.Entry(row_f, textvariable=var, width=7, justify='center')
            entry.pack(side=tk.LEFT, padx=2)
            entry.bind('<Return>', lambda e, k=key: self._on_entry_commit(k))
            entry.bind('<FocusOut>', lambda e, k=key: self._on_entry_commit(k))
            self.adjust_entries[key] = entry
            ttk.Button(row_f, text="▶", width=3,
                       command=lambda k=key: self.adjust_box(k, 1)).pack(side=tk.LEFT, padx=1)

        # 步长选择
        step_f = ttk.Frame(adj_lf)
        step_f.pack(fill=tk.X, padx=4, pady=3)
        ttk.Label(step_f, text="微调步长:").pack(side=tk.LEFT)
        self.step_var = tk.IntVar(value=1)
        ttk.Spinbox(step_f, from_=1, to=100, textvariable=self.step_var,
                    width=6).pack(side=tk.LEFT, padx=6)

        # ── 操作按钮 ──
        btn_f = ttk.Frame(right_frame)
        btn_f.pack(fill=tk.X, padx=3, pady=4)
        ttk.Button(btn_f, text="下一个", command=self.next_annotation).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_f, text="确定 ✓", command=self.confirm_image).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_f, text="删除选中", command=self.delete_annotation).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        # 清空当前标注(撤销当前正在画的框)
        ttk.Button(btn_f, text="撤销框", command=self.clear_current_bbox).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # ── 保存目录 ──
        save_f = ttk.Frame(right_frame)
        save_f.pack(fill=tk.X, padx=3, pady=2)
        ttk.Label(save_f, text="保存至:").pack(side=tk.LEFT)
        self.save_dir_label = ttk.Label(save_f, text="(图片同目录)", foreground='gray')
        self.save_dir_label.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(save_f, text="选择...", command=self.select_save_dir).pack(side=tk.RIGHT)

        # ── 底部: 标注结果列表 ──
        bottom_frame = ttk.LabelFrame(self.root, text="标注结果")
        bottom_frame.pack(fill=tk.BOTH, expand=False, padx=3, pady=2, side=tk.BOTTOM)

        columns = ('序号', 'ID', '类别', 'x1', 'y1', 'x2', 'y2')
        self.tree = ttk.Treeview(bottom_frame, columns=columns, show='headings',
                                 height=6, selectmode='browse')
        col_widths = [50, 40, 80, 70, 70, 70, 70]
        for col, w in zip(columns, col_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor='center')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_sb = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_sb.set)
        tree_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # 双击 Treeview 行 → 选中并加载到画布进行修改
        self.tree.bind('<Double-1>', self._on_tree_double_click)

        # ── 状态栏 ──
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=(6, 6),
                               font=('Microsoft YaHei UI', 19))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ══════════════════════════════════════════════════════════
    # 事件绑定
    # ══════════════════════════════════════════════════════════
    def _bind_events(self):
        # 鼠标画框
        self.canvas.bind('<ButtonPress-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)

        # 滚轮缩放
        self.canvas.bind('<Control-MouseWheel>', self._on_ctrl_wheel)
        # Windows 上 MouseWheel, Linux 上 Button-4/5
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.canvas.bind('<Button-4>', self._on_mouse_wheel)
        self.canvas.bind('<Button-5>', self._on_mouse_wheel)

        # 键盘快捷键
        self.root.bind('<Control-plus>', lambda e: self.zoom_in())
        self.root.bind('<Control-minus>', lambda e: self.zoom_out())
        self.root.bind('<Control-0>', lambda e: self.zoom_100())
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Escape>', lambda e: self.clear_current_bbox())

    # ══════════════════════════════════════════════════════════
    # 文件 / 图片操作
    # ══════════════════════════════════════════════════════════
    def open_folder(self):
        """打开图片文件夹"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        self.image_folder = folder
        # 扫描图片
        self.image_list = []
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                self.image_list.append(os.path.join(folder, fname))
        if not self.image_list:
            messagebox.showinfo("提示", "所选文件夹中没有找到图片文件。")
            return
        self.current_image_idx = 0
        self._load_and_display()

    def _load_and_display(self):
        """加载当前索引的图片并显示，自动搜索同名txt"""
        if self.current_image_idx < 0 or self.current_image_idx >= len(self.image_list):
            return

        img_path = self.image_list[self.current_image_idx]
        try:
            self.pil_image = Image.open(img_path)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片:\n{img_path}\n\n{e}")
            return

        # 保持当前缩放比例（首次加载时用fit_to_window初始化）
        if self.zoom_scale <= 0.05:
            self.fit_to_window()

        # 清空标注
        self.annotations.clear()
        self.clear_current_bbox()
        self._refresh_tree()

        # 更新导航信息
        self._update_nav_label()

        # 搜索同名txt
        txt_path = self._get_txt_path()
        if os.path.exists(txt_path):
            self._load_annotations_from_file(txt_path)
            self._update_status(f"已加载已有标注: {os.path.basename(txt_path)}  ({len(self.annotations)} 条)")
        else:
            self._update_status(f"已加载图片: {os.path.basename(img_path)}  "
                                f"({self.pil_image.width}×{self.pil_image.height})")

        # 重绘（含已保存标注框）
        self._redraw_all()

    def _get_txt_path(self) -> str:
        """获取当前图片对应的txt标注文件路径"""
        if self.current_image_idx < 0:
            return ""
        img_path = self.image_list[self.current_image_idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = self.save_directory if self.save_directory else os.path.dirname(img_path)
        return os.path.join(save_dir, f"{base}.txt")

    # ══════════════════════════════════════════════════════════
    # 标注文件读写
    # ══════════════════════════════════════════════════════════
    def _save_annotations_to_file(self, filepath: str):
        """将当前标注写入txt文件（JSON结构体，{}分级嵌套）"""
        if self.pil_image is None:
            return
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data = [ann.to_dict() for ann in self.annotations]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("保存失败", f"无法写入标注文件:\n{filepath}\n\n{e}")

    def _load_annotations_from_file(self, filepath: str):
        """从txt文件加载标注（JSON结构体 → Annotation列表）"""
        self.annotations.clear()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    ann = Annotation.from_dict(item)
                    if ann:
                        self.annotations.append(ann)
        except Exception as e:
            messagebox.showwarning("读取标注", f"读取标注文件出错:\n{filepath}\n\n{e}")
        self._refresh_tree()

    # ══════════════════════════════════════════════════════════
    # 图像显示 / 缩放
    # ══════════════════════════════════════════════════════════
    def _render_image(self):
        """根据当前缩放比例生成 tk_image 并更新 Canvas"""
        if self.pil_image is None:
            return

        w, h = self.pil_image.size
        new_w = max(1, int(w * self.zoom_scale))
        new_h = max(1, int(h * self.zoom_scale))

        # 使用高质量重采样
        resized = self.pil_image.resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        # 更新或创建Canvas图像
        if self.canvas_img_id is not None:
            self.canvas.itemconfig(self.canvas_img_id, image=self.tk_image)
        else:
            self.canvas_img_id = self.canvas.create_image(0, 0, anchor='nw',
                                                          image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

        # 更新缩放标签
        self.zoom_label.config(text=f"{int(self.zoom_scale * 100)}%")

    def _redraw_all(self, keep_scroll: bool = True):
        """重绘图像和所有标注框"""
        # 保存当前滚动位置（相对于图像的比例）
        sx_prop = sy_prop = 0.0
        if keep_scroll and self.canvas_img_id is not None:
            sx_prop = self.canvas.canvasx(0) / max(1, self.canvas.bbox('all')[2])
            sy_prop = self.canvas.canvasy(0) / max(1, self.canvas.bbox('all')[3])

        self.canvas.delete('all')
        self.canvas_img_id = None
        self.preview_rect_id = None
        self.saved_rect_ids.clear()

        if self.pil_image is None:
            return

        self._render_image()

        # 恢复滚动位置
        if keep_scroll:
            region = self.canvas.bbox('all')
            if region:
                self.canvas.xview_moveto(sx_prop)
                self.canvas.yview_moveto(sy_prop)

        # 绘制已保存的标注框（蓝色）
        for ann in self.annotations:
            c_x1, c_y1 = self._to_canvas(ann.x1, ann.y1)
            c_x2, c_y2 = self._to_canvas(ann.x2, ann.y2)
            rid = self.canvas.create_rectangle(
                c_x1, c_y1, c_x2, c_y2,
                outline='#00aaff', width=2, tags='saved_bbox'
            )
            # 在框左上角显示类别标签
            self.canvas.create_text(
                c_x1 + 2, c_y1 - 10 if c_y1 > 15 else c_y1 + 12,
                text=ann.category, anchor='w', fill='#00aaff',
                font=('Microsoft YaHei UI', 20, 'bold'), tags='saved_bbox'
            )
            self.saved_rect_ids.append(rid)

        # 绘制当前正在画的预览框（红色）
        if self.current_bbox is not None and self.current_category is not None:
            self._draw_preview_rect()

        # 确保图像在底层
        if self.canvas_img_id is not None:
            self.canvas.tag_lower(self.canvas_img_id)

    def _draw_preview_rect(self):
        """绘制当前预览红色框"""
        if self.preview_rect_id is not None:
            self.canvas.delete(self.preview_rect_id)
            # 同时删除预览标签
            self.canvas.delete('preview_label')

        if self.current_bbox is None or self.current_category is None:
            return

        c_x1, c_y1 = self._to_canvas(self.current_bbox[0], self.current_bbox[1])
        c_x2, c_y2 = self._to_canvas(self.current_bbox[2], self.current_bbox[3])
        self.preview_rect_id = self.canvas.create_rectangle(
            c_x1, c_y1, c_x2, c_y2,
            outline='red', width=2, dash=(4, 2)
        )
        self.canvas.create_text(
            c_x1 + 2, c_y1 - 10 if c_y1 > 15 else c_y1 + 12,
            text=self.current_category, anchor='w', fill='red',
            font=('Microsoft YaHei UI', 20, 'bold'), tags='preview_label'
        )

    def _update_bbox_display(self):
        """将current_bbox的值同步到微调输入框"""
        if self.current_bbox is not None:
            for key, val in zip(['x1', 'y1', 'x2', 'y2'], self.current_bbox):
                self.adjust_vars[key].set(val)
        else:
            for key in ['x1', 'y1', 'x2', 'y2']:
                self.adjust_vars[key].set(0)

    def zoom_in(self):
        """放大"""
        if self.pil_image is None:
            return
        self.zoom_scale = min(self.zoom_scale * 1.25, 20.0)
        self._redraw_all()

    def zoom_out(self):
        """缩小"""
        if self.pil_image is None:
            return
        self.zoom_scale = max(self.zoom_scale / 1.25, 0.05)
        self._redraw_all()

    def zoom_100(self):
        """原始大小"""
        if self.pil_image is None:
            return
        self.zoom_scale = 1.0
        self._redraw_all()

    def fit_to_window(self, retry: int = 0):
        """适应窗口"""
        if self.pil_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if (cw < 10 or ch < 10) and retry < 10:
            # Canvas 尚未布局完成，延迟重试
            self.root.after(100, lambda: self.fit_to_window(retry + 1))
            return
        iw, ih = self.pil_image.size
        self.zoom_scale = min(cw / max(iw, 1), ch / max(ih, 1), 1.0)
        self._redraw_all(keep_scroll=False)

    # ══════════════════════════════════════════════════════════
    # 坐标转换（图像始终位于Canvas坐标(0,0)，缩放原点即图像原点）
    # ══════════════════════════════════════════════════════════
    def _to_canvas(self, img_x: int, img_y: int) -> Tuple[float, float]:
        """图像坐标 → Canvas坐标"""
        return (img_x * self.zoom_scale, img_y * self.zoom_scale)

    def _canvas_to_image(self, canvas_x: float, canvas_y: float) -> Tuple[int, int]:
        """Canvas坐标 → 图像坐标
        输入应为 canvas.canvasx/canvasy 转换后的Canvas坐标
        """
        return (int(round(canvas_x / self.zoom_scale)),
                int(round(canvas_y / self.zoom_scale)))

    # ══════════════════════════════════════════════════════════
    # 类别选择
    # ══════════════════════════════════════════════════════════
    def select_category(self, category: str):
        """选择标注类别"""
        self.current_category = category
        self.cat_hint.config(text=f"当前类别: {category}", foreground='#007acc')
        # 高亮选中的按钮
        for name, btn in self._cat_btn_ref.items():
            if name == category:
                btn.state(['pressed'])
            else:
                btn.state(['!pressed'])
        self._update_status(f"已选择类别「{category}」— 请在图像上拖拽画框")

    # ══════════════════════════════════════════════════════════
    # 鼠标画框交互
    # ══════════════════════════════════════════════════════════
    def _on_mouse_down(self, event):
        if self.pil_image is None or self.current_category is None:
            if self.current_category is None:
                self._update_status("请先选择一个标注类别")
            return
        # 记录起始点（Canvas坐标）
        self.drawing = True
        self.drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.configure(cursor='crosshair')

    def _on_mouse_drag(self, event):
        if not self.drawing:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        sx, sy = self.drag_start

        # 计算图像坐标（绑定在图像范围内）
        img_w, img_h = self.pil_image.size
        i_x1, i_y1 = self._canvas_to_image(sx, sy)
        i_x2, i_y2 = self._canvas_to_image(cx, cy)

        # 钳位到图像范围
        i_x1 = max(0, min(i_x1, img_w))
        i_y1 = max(0, min(i_y1, img_h))
        i_x2 = max(0, min(i_x2, img_w))
        i_y2 = max(0, min(i_y2, img_h))

        self.current_bbox = [i_x1, i_y1, i_x2, i_y2]
        self._draw_preview_rect()
        self._update_bbox_display()

    def _on_mouse_up(self, event):
        if not self.drawing:
            return
        self.drawing = False
        self.canvas.configure(cursor='crosshair')

        # 如果框太小（<3像素），视为无效
        if self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            if abs(x2 - x1) < 3 and abs(y2 - y1) < 3:
                self.clear_current_bbox()
                self._update_status("框太小，已忽略。请重新绘制。")
                return

        if self.current_bbox is not None:
            # 规范化为 x1<x2, y1<y2
            x1, y1, x2, y2 = self.current_bbox
            self.current_bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            self._draw_preview_rect()
            self._update_bbox_display()
            self._update_status(
                f"框已绘制: ({self.current_bbox[0]}, {self.current_bbox[1]}) "
                f"→ ({self.current_bbox[2]}, {self.current_bbox[3]}) — "
                f"可微调或点击「下一个」存储"
            )

    def _on_mouse_wheel(self, event):
        """鼠标滚轮缩放 (无Ctrl时也缩放，方便操作)"""
        if self.pil_image is None:
            return
        # Linux: Button-4/5, Windows/Mac: MouseWheel
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.zoom_in()
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom_out()

    def _on_ctrl_wheel(self, event):
        """Ctrl+滚轮缩放（保留兼容）"""
        self._on_mouse_wheel(event)

    # ══════════════════════════════════════════════════════════
    # 微调框
    # ══════════════════════════════════════════════════════════
    def adjust_box(self, key: str, delta: int):
        """微调框的某个参数"""
        if self.current_bbox is None or self.pil_image is None:
            self._update_status("请先选择类别并画框")
            return

        step = self.step_var.get() * delta
        idx_map = {'x1': 0, 'y1': 1, 'x2': 2, 'y2': 3}
        idx = idx_map[key]

        new_val = self.current_bbox[idx] + step
        # 钳位
        img_w, img_h = self.pil_image.size
        max_vals = [img_w, img_h, img_w, img_h]
        new_val = max(0, min(new_val, max_vals[idx]))

        # x1 < x2, y1 < y2 约束
        if idx == 0:  # x1
            new_val = min(new_val, self.current_bbox[2] - 1)
        elif idx == 1:  # y1
            new_val = min(new_val, self.current_bbox[3] - 1)
        elif idx == 2:  # x2
            new_val = max(new_val, self.current_bbox[0] + 1)
        elif idx == 3:  # y2
            new_val = max(new_val, self.current_bbox[1] + 1)

        self.current_bbox[idx] = new_val
        self._draw_preview_rect()
        self._update_bbox_display()

    def _on_entry_commit(self, key: str):
        """用户手动输入框参数后回车/失焦"""
        if self.current_bbox is None or self.pil_image is None:
            return
        try:
            val = self.adjust_vars[key].get()
        except tk.TclError:
            return
        idx_map = {'x1': 0, 'y1': 1, 'x2': 2, 'y2': 3}
        idx = idx_map[key]
        img_w, img_h = self.pil_image.size
        max_vals = [img_w, img_h, img_w, img_h]
        val = max(0, min(val, max_vals[idx]))

        # 约束
        if idx == 0:
            val = min(val, self.current_bbox[2] - 1)
        elif idx == 1:
            val = min(val, self.current_bbox[3] - 1)
        elif idx == 2:
            val = max(val, self.current_bbox[0] + 1)
        elif idx == 3:
            val = max(val, self.current_bbox[1] + 1)

        self.current_bbox[idx] = val
        self._draw_preview_rect()
        self._update_bbox_display()

    # ══════════════════════════════════════════════════════════
    # 标注操作
    # ══════════════════════════════════════════════════════════
    def next_annotation(self):
        """存储当前标注，准备下一个"""
        if self.current_bbox is None or self.current_category is None:
            self._update_status("请先选择类别并画框")
            return

        # 检查是否与已有标注重复
        x1, y1, x2, y2 = self.current_bbox
        for ann in self.annotations:
            if ann.category == self.current_category and \
               ann.x1 == x1 and ann.y1 == y1 and \
               ann.x2 == x2 and ann.y2 == y2:
                if not messagebox.askyesno("重复标注",
                                           f"类别「{self.current_category}」已存在完全相同的标注框。\n"
                                           "是否仍要添加？"):
                    return

        ann = Annotation(
            category=self.current_category,
            x1=x1, y1=y1, x2=x2, y2=y2
        )
        self.annotations.append(ann)
        self._refresh_tree()
        self._redraw_all()

        # 重置当前标注状态
        self.current_bbox = None
        self._update_bbox_display()
        self._draw_preview_rect()
        self._update_status(
            f"已存储「{ann.category}」标注 ({len(self.annotations)} 条) — "
            f"可继续选择类别标注"
        )

    def confirm_image(self):
        """确认当前图片标注完成，保存到txt"""
        if self.pil_image is None:
            return
        # 如果有未提交的当前框，询问是否一并保存
        if self.current_bbox is not None and self.current_category is not None:
            if messagebox.askyesno("未保存的标注",
                                   "当前还有未提交的标注框，是否一并保存？\n"
                                   f"类别: {self.current_category}\n"
                                   f"位置: {self.current_bbox}"):
                self.next_annotation()  # 先存入列表

        if not self.annotations:
            if not messagebox.askyesno("确认", "当前图片没有任何标注，确定保存空文件？"):
                return

        txt_path = self._get_txt_path()
        self._save_annotations_to_file(txt_path)
        self._update_status(f"✓ 已保存: {os.path.basename(txt_path)}  ({len(self.annotations)} 条标注)")
        messagebox.showinfo("保存成功",
                            f"标注已保存到:\n{txt_path}\n\n共 {len(self.annotations)} 条标注")

    def delete_annotation(self):
        """删除选中的标注"""
        selection = self.tree.selection()
        if not selection:
            self._update_status("请在标注结果列表中选中要删除的行")
            return
        item = selection[0]
        idx = int(self.tree.index(item))
        if 0 <= idx < len(self.annotations):
            ann = self.annotations[idx]
            if messagebox.askyesno("确认删除", f"确定删除标注:\n类别: {ann.category}\n"
                                              f"位置: ({ann.x1},{ann.y1}) → ({ann.x2},{ann.y2}) ?"):
                del self.annotations[idx]
                self._refresh_tree()
                self._redraw_all()
                self._update_status(f"已删除标注，当前共 {len(self.annotations)} 条")

    def clear_current_bbox(self):
        """清除当前正在画的预览框"""
        self.current_bbox = None
        self._update_bbox_display()
        self._draw_preview_rect()
        # 清除类别按钮选中状态
        for btn in self.cat_buttons:
            btn.state(['!pressed'])
        self.current_category = None
        self.cat_hint.config(text="当前类别: 未选择", foreground='#888')
        self._update_status("已撤销当前标注框")

    def _refresh_tree(self):
        """刷新Treeview显示"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, ann in enumerate(self.annotations, 1):
            cid = CATEGORY_TO_ID.get(ann.category, -1)
            self.tree.insert('', 'end', values=(i, cid, ann.category, ann.x1, ann.y1, ann.x2, ann.y2))

    def _on_tree_double_click(self, event):
        """双击Treeview行: 加载该标注到画布进行修改"""
        selection = self.tree.selection()
        if not selection:
            return
        item = selection[0]
        idx = int(self.tree.index(item))
        if 0 <= idx < len(self.annotations):
            ann = self.annotations[idx]
            # 将标注加载为当前编辑框
            self.current_category = ann.category
            self.cat_hint.config(text=f"当前类别: {ann.category}", foreground='#007acc')
            # 高亮按钮
            for name, btn in self._cat_btn_ref.items():
                if name == ann.category:
                    btn.state(['pressed'])
                else:
                    btn.state(['!pressed'])
            self.current_bbox = [ann.x1, ann.y1, ann.x2, ann.y2]
            self._draw_preview_rect()
            self._update_bbox_display()
            # 从列表中移除（修改后需重新"下一个"提交）
            del self.annotations[idx]
            self._refresh_tree()
            self._redraw_all()
            self._update_status(
                f"正在修改标注「{ann.category}」— 修改后请点击「下一个」重新存储"
            )

    # ══════════════════════════════════════════════════════════
    # 导航
    # ══════════════════════════════════════════════════════════
    def prev_image(self):
        """上一张图片"""
        if not self.image_list:
            return
        if self.current_image_idx <= 0:
            self._update_status("已经是第一张")
            return
        self._auto_save_if_needed()
        self.current_image_idx -= 1
        self._load_and_display()

    def next_image(self):
        """下一张图片"""
        if not self.image_list:
            return
        if self.current_image_idx >= len(self.image_list) - 1:
            self._update_status("已经是最后一张")
            return
        self._auto_save_if_needed()
        self.current_image_idx += 1
        self._load_and_display()

    def _auto_save_if_needed(self):
        """切换图片前自动保存（如果标注已通过「确定」保存则跳过，否则询问并保存）"""
        if self.pil_image is None:
            return
        # 如果有未提交的当前框，提示
        if self.current_bbox is not None and self.current_category is not None:
            if messagebox.askyesno("未保存的标注",
                                   "当前有未提交的标注框，切换前是否保存？"):
                self.next_annotation()

        if self.annotations:
            txt_path = self._get_txt_path()
            if not os.path.exists(txt_path):
                if messagebox.askyesno("自动保存",
                                       f"当前图片有 {len(self.annotations)} 条标注，"
                                       "是否保存后再切换？"):
                    self._save_annotations_to_file(txt_path)
                    self._update_status(f"已自动保存: {os.path.basename(txt_path)}")
            else:
                # txt已存在（可能是加载的或之前保存的），检查内容是否一致
                self._save_annotations_to_file(txt_path)

    def _update_nav_label(self):
        """更新导航标签"""
        if not self.image_list:
            self.image_label.config(text="未加载图片")
            return
        total = len(self.image_list)
        cur = self.current_image_idx + 1
        fname = os.path.basename(self.image_list[self.current_image_idx])
        self.image_label.config(text=f"[{cur}/{total}]  {fname}")

    # ══════════════════════════════════════════════════════════
    # 保存目录
    # ══════════════════════════════════════════════════════════
    def select_save_dir(self):
        """选择自定义保存目录"""
        d = filedialog.askdirectory(title="选择标注保存目录")
        if d:
            self.save_directory = d
            self.save_dir_label.config(text=d, foreground='#000')
        else:
            self.save_directory = ""
            self.save_dir_label.config(text="(图片同目录)", foreground='gray')

    # ══════════════════════════════════════════════════════════
    # 杂项
    # ══════════════════════════════════════════════════════════
    def _update_status(self, text: str):
        """更新状态栏"""
        self.status_var.set(text)

    def _show_about(self):
        messagebox.showinfo("关于",
                            "FOD 图像标注工具 v1.2\n\n"
                            "用于外来物碎片(FOD)图像的 Bounding Box 标注。\n"
                            "支持12种类别（金属/橡胶/塑料/木头/纸板/\n"
                            "织物/动物/玻璃/石头/泡沫/油渍/水），\n"
                            "缩放标注，结果存为JSON结构体格式txt。\n"
                            "类别以0-11整数编号存储，{}分级嵌套。\n\n"
                            "快捷键:\n"
                            "  左右箭头: 切换图片\n"
                            "  Ctrl+/-: 缩放\n"
                            "  Esc: 撤销当前框\n"
                            "  滚轮: 缩放")

    def _on_close(self):
        """关闭窗口时的处理"""
        if self.annotations:
            if messagebox.askyesno("退出确认",
                                   "当前图片有未保存的标注，是否保存后退出？"):
                txt_path = self._get_txt_path()
                # 先提交未保存的当前框
                if self.current_bbox is not None and self.current_category is not None:
                    if messagebox.askyesno("未保存的标注",
                                           "还有未提交的标注框，是否一并保存？"):
                        self.next_annotation()
                self._save_annotations_to_file(txt_path)
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()

    # 设置DPI感知 (Windows高分屏)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # ── 全局字体放大 ──
    FONT_BASE = ('Microsoft YaHei UI', 20)
    FONT_BOLD = ('Microsoft YaHei UI', 20, 'bold')
    FONT_SMALL = ('Microsoft YaHei UI', 19)
    FONT_MENU = ('Microsoft YaHei UI', 23)       # 菜单/对话框 +6

    # 设置 tk 根默认字体（影响对话框等）
    root.option_add('*Font', FONT_MENU)
    root.option_add('*Dialog.msg.font', FONT_MENU)

    style = ttk.Style()
    available = style.theme_names()
    for preferred in ('vista', 'clam', 'alt', 'default'):
        if preferred in available:
            style.theme_use(preferred)
            break

    style.configure('TButton', font=FONT_BASE)
    style.configure('TLabel', font=FONT_BASE)
    style.configure('TLabelframe.Label', font=FONT_BOLD)
    style.configure('TEntry', font=FONT_BASE)
    style.configure('TSpinbox', font=FONT_BASE)
    style.configure('Treeview', font=FONT_BASE, rowheight=48)
    style.configure('Treeview.Heading', font=FONT_BOLD)
    # 菜单字体 +6
    style.configure('TMenu', font=FONT_MENU)
    style.configure('TStatusbar.TLabel', font=FONT_SMALL)

    app = FODAnnotationTool(root)
    root.mainloop()


if __name__ == '__main__':
    main()
