import pytmx


def create_construction(_id: int or str, _no: int, _init_pos: tuple, _init_size: tuple):
    return {
        "_id": _id,
        "_no": _no,
        "_init_pos": _init_pos,
        "_init_size": _init_size
    }


# Map 讀取地圖資料
class TiledMap:
    def __init__(self, filepath: str):
        tm = pytmx.TiledMap(filepath)
        self.tile_width = tm.tilewidth
        self.tile_height = tm.tileheight
        self.width = tm.width
        self.height = tm.height
        self.map_width = self.tile_width * self.width
        self.map_height = self.tile_height * self.height
        self.tmx_data = tm
        self._is_record = False
        self.empty_pos_list = []
        self.all_pos_list = []
        self.empty_quadrant_1_pos_list = []
        self.empty_quadrant_2_pos_list = []
        self.empty_quadrant_3_pos_list = []
        self.empty_quadrant_4_pos_list = []

    def create_init_obj(self, img_no: int, class_name, **kwargs) -> dict:
        obj_no = 0
        for layer in self.tmx_data.visible_layers:
            for x, y, gid, in layer:
                if isinstance(layer, pytmx.TiledTileLayer):
                    if gid:  # 0代表空格，無圖塊
                        if layer.parent.tiledgidmap[gid] == img_no:
                            img_id = layer.parent.tiledgidmap[gid]
                            obj_no += 1
                            img_info = {"_id": img_id, "_no": obj_no,
                                        "_init_pos": (x * self.tile_width, y * self.tile_height),
                                        "_init_size": (self.tile_width, self.tile_height)}
                            obj = class_name(img_info, **kwargs)
                            return obj

    def create_init_obj_list(self, img_no: list or str, class_name, **kwargs) -> list:
        if type(img_no) != list:
            img_no = list(map(int, [img_no]))
        obj_result = []
        obj_no = 0
        for layer in self.tmx_data.visible_layers:
            for x, y, gid, in layer:
                if isinstance(layer, pytmx.TiledTileLayer):
                    pos = (x * self.tile_width, y * self.tile_height)
                    if not self._is_record:
                        self.all_pos_list.append(pos)
                    if not self._is_record and not gid:  # 0代表空格，無圖塊
                        self.empty_pos_list.append(pos)
                        if pos[0] >= self.map_width // 2 and pos[1] < self.map_height // 2:
                            self.empty_quadrant_1_pos_list.append(pos)
                        elif pos[0] < self.map_width // 2 and pos[1] < self.map_height // 2:
                            self.empty_quadrant_2_pos_list.append(pos)
                        elif pos[0] < self.map_width // 2 and pos[1] >= self.map_height // 2:
                            self.empty_quadrant_3_pos_list.append(pos)
                        else:
                            self.empty_quadrant_4_pos_list.append(pos)
                    elif gid:
                        if layer.parent.tiledgidmap[gid] in img_no:
                            img_id = layer.parent.tiledgidmap[gid]
                            obj_no += 1
                            img_info = {"_id": img_id, "_no": obj_no,
                                        "_init_pos": pos,
                                        "_init_size": (self.tile_width, self.tile_height)}
                            obj_result.append(class_name(img_info, **kwargs))
        self._is_record = True
        return obj_result
