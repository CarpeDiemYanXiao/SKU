from abc import ABC
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class Snapshots(ABC):
    axs_map = {}
    points = {
        # name: {x: [], y: []}
    }

    def __init__(self, name_x: dict, y_locators: dict = {}) -> None:
        super().__init__()
        self.name_x = name_x
        self.y_locators = y_locators

    def __init_plt(self):
        if self.axs_map and len(self.axs_map) != 0:
            return
        plt.ion()
        self.fig, self.axs = plt.subplots(len(self.name_x), 1, figsize=(18, 10))
        if len(self.name_x) == 1:
            self.axs = [self.axs]
        self.axs_map = {name: axis for name, axis in zip(self.name_x, self.axs)}

    def record(self, part: str, name: str, x, y):
        if part + "::" + name not in self.points:
            self.points[part + "::" + name] = {"x": [], "y": []}
        self.points[part + "::" + name]["x"].append(x)
        self.points[part + "::" + name]["y"].append(y)

    def re_draw_map(self, name, info: dict):
        self.__init_plt()
        axis = self.axs_map[name]
        axis.cla()
        axis.grid(True)
        axis.set_xlim(0, len(info) + 1)

        x_labels = list(info.keys())
        x_values = list(info.values())
        axis.bar(x_labels, x_values)
        plt.draw()

    def draw_line(self):
        self.__init_plt()
        for part, axis in self.axs_map.items():
            axis.cla()
            axis.grid(True)
            # axis.set_xlim(0, self.name_x[part] + 1)
            axis.set_xlim(right=10)

            for name, point in self.points.items():
                # name 的结构是 part_name::metric_name
                current_part, metric_name = name.split("::")
                # 只在当前 part 下画图
                if current_part == part:
                    axis.plot(point["x"], point["y"], marker="o", linestyle="-", label=metric_name)

            major_interval = self.y_locators.get(part, 0.5)
            axis.yaxis.set_major_locator(MultipleLocator(major_interval))
            axis.set_title(part)
            axis.legend()

        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)

    # def draw_table(self, only_name: str | None = None):
    def draw_table(self, only_name: str):
        self.__init_plt()
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        ax.cla()
        ax.axis("off")

        data = [[name] + point["y"] for name, point in self.points.items() if not only_name or only_name + "::" in name]
        columns = ["name"] + [str(i) for i in range(len(data[0]) - 1)]
        table = ax.table(cellText=data, colLabels=columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(list(range(len(columns))))
        plt.draw()
