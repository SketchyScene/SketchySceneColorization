# -*- coding: utf-8 -*-
import data_generation_util

dictWeather = {
    'sun': 'It\'s a sunny day.',
    'cloud': 'It\'s a cloudy day.',
    'moon': 'It\'s a moonlit night.'
}


class ImageToText(object):
    def __init__(self, items):
        """
        Args:
            items: list?elements are @code{Item}
        """
        super(ImageToText, self).__init__()
        self.items = items
        self.sorted_indices = []  # the sorted indices of input instances, like: [7, 6, 2, 0, 10, 12, 5, 9, 11, 8, 13]
        self.sen_instIdx_map = []  # all instance idx of each sentence, -1 means no instance related
        # like: [[7, 6, 2, 0, 10], [12], [5], [9], [-1], [11], [8], [13], [-1]]

    def get_weather_summary(self):
        """
            1. sunny day/ moonlit night first
            2. cloudy day second
        """
        is_cloudy = False
        for item in self.items:
            cate = item.category
            if cate == 'sun' or cate == 'moon':
                self.sen_instIdx_map.append([-1])
                return dictWeather[cate]
            elif cate == 'cloud':
                is_cloudy = True
        if is_cloudy:
            self.sen_instIdx_map.append([-1])
            return dictWeather['cloud']
        else:
            return ""

    def get_weather_singles(self):
        """
            1. sun or moon first: 'There is a sun in the sky.'
            2. cloud second: 'A/Two/Many cloud is/are floating in the air.'
            3. star third
        """
        cloud_ids = []
        star_ids = []
        sun_ids = []
        moon_ids = []
        for item in self.items:
            cate = item.category
            index = item.id
            if cate == "cloud":
                cloud_ids.append(index)
            elif cate == "star":
                star_ids.append(index)
            elif cate == "sun":
                sun_ids.append(index)
            elif cate == "moon":
                moon_ids.append(index)

        texts = []

        ## 1. describe sun or moon
        if len(sun_ids) == 1:
            texts.append('the sun in the sky.')
            self.sen_instIdx_map.append(sun_ids)
            self.sorted_indices.extend(sun_ids)

            texts.append('the sun.')
            self.sen_instIdx_map.append(sun_ids)
            self.sorted_indices.extend(sun_ids)

        elif len(sun_ids) >= 2:

            texts.append('the suns.')
            self.sen_instIdx_map.append(sun_ids)
            self.sorted_indices.extend(sun_ids)

            if len(sun_ids) == 2:
                texts.append('both the suns.')
                self.sen_instIdx_map.append(sun_ids)
                self.sorted_indices.extend(sun_ids)
            else:
                texts.append('all the suns.')
                self.sen_instIdx_map.append(sun_ids)
                self.sorted_indices.extend(sun_ids)

        if len(moon_ids) == 1:
            texts.append('the moon in the sky.')
            self.sen_instIdx_map.append(moon_ids)
            self.sorted_indices.extend(moon_ids)

            texts.append('the moon.')
            self.sen_instIdx_map.append(moon_ids)
            self.sorted_indices.extend(moon_ids)

        elif len(moon_ids) >= 2:

            texts.append('the moons.')
            self.sen_instIdx_map.append(moon_ids)
            self.sorted_indices.extend(moon_ids)

            if len(moon_ids) == 2:
                texts.append('both the moons.')
                self.sen_instIdx_map.append(moon_ids)
                self.sorted_indices.extend(moon_ids)
            else:
                texts.append('all the moons.')
                self.sen_instIdx_map.append(moon_ids)
                self.sorted_indices.extend(moon_ids)

        ## 2. describe cloud
        if len(cloud_ids) == 1:
            texts.append('the cloud in the sky.')
            self.sen_instIdx_map.append(cloud_ids)
            self.sorted_indices.extend(cloud_ids)

            texts.append('the cloud.')
            self.sen_instIdx_map.append(cloud_ids)
            self.sorted_indices.extend(cloud_ids)

        elif len(cloud_ids) >= 2:

            texts.append('the clouds.')
            self.sen_instIdx_map.append(cloud_ids)
            self.sorted_indices.extend(cloud_ids)

            if len(cloud_ids) == 2:
                texts.append('both the clouds.')
                self.sen_instIdx_map.append(cloud_ids)
                self.sorted_indices.extend(cloud_ids)
            else:
                texts.append('all the clouds.')
                self.sen_instIdx_map.append(cloud_ids)
                self.sorted_indices.extend(cloud_ids)

        ## 3. describe star
        if len(star_ids) == 1:
            texts.append('the star in the sky.')
            self.sen_instIdx_map.append(star_ids)
            self.sorted_indices.extend(star_ids)

            texts.append('the star.')
            self.sen_instIdx_map.append(star_ids)
            self.sorted_indices.extend(star_ids)

        elif len(star_ids) >= 2:
            texts.append('the stars in the sky.')
            self.sen_instIdx_map.append(star_ids)
            self.sorted_indices.extend(star_ids)

            texts.append('the stars.')
            self.sen_instIdx_map.append(star_ids)
            self.sorted_indices.extend(star_ids)

            if len(star_ids) == 2:
                texts.append('both the stars.')
                self.sen_instIdx_map.append(star_ids)
                self.sorted_indices.extend(star_ids)
            else:
                texts.append('all the stars.')
                self.sen_instIdx_map.append(star_ids)
                self.sorted_indices.extend(star_ids)

        weather_singles_texts = " ".join(texts)
        return weather_singles_texts

    def get_ground_items(self):
        """
            1. unmovable objects first: house, bench, bus, car
            2. tree second
            3. movable objects(animals) lastly

            PS. Objects in same category, which are closed to each other, will be regarded to be a group.
                Objects in group will be described together, otherwise indivually

           """
        categories_on_ground = data_generation_util.CATEGORIES_UNMOVABLE + data_generation_util.CATEGORIES_TREE + data_generation_util.CATEGORIES_MOVABLE
        ground_items = [item for item in self.items if item.category in categories_on_ground]

        unmovable, trees, movable = \
            data_generation_util.ItemCollection.get_collections(ground_items)

        # print("unmovable:", (unmovable.get_description())["des"])
        # print("trees:", (trees.get_description())["des"])
        # print("movable:", (movable.get_description())["des"])

        unmovable_res = unmovable.get_description()
        trees_res = trees.get_description()
        movable_res = movable.get_description()

        descriptions = [unmovable_res["des"], trees_res["des"], movable_res["des"]]
        self.sorted_indices.extend(
            unmovable_res["sorted_indices"] + trees_res["sorted_indices"] + movable_res["sorted_indices"])

        sen_instIdx_map_list = [unmovable_res["sen_instIdx_map"],
                                trees_res["sen_instIdx_map"],
                                movable_res["sen_instIdx_map"]]
        for indexes in sen_instIdx_map_list:
            self.sen_instIdx_map.extend(indexes)

        ground_items_text = " ".join([des for des in descriptions if des != ""])
        return ground_items_text

    def get_grass_road_text(self):
        """

        """
        grass_ids = []
        road_ids = []
        for item in self.items:
            cate = item.category
            index = item.id
            if cate == "grass":
                grass_ids.append(index)
            elif cate == "road":
                road_ids.append(index)

        texts = []

        ## 1. describe grass
        if len(grass_ids) == 1:
            texts.append('the grass.')
            self.sen_instIdx_map.append(grass_ids)
            self.sorted_indices.extend(grass_ids)

        elif len(grass_ids) >= 2:
            texts.append('the grasses.')
            self.sen_instIdx_map.append(grass_ids)
            self.sorted_indices.extend(grass_ids)

            if len(grass_ids) == 2:
                texts.append('both the grass.')
                self.sen_instIdx_map.append(grass_ids)
                self.sorted_indices.extend(grass_ids)
            else:
                texts.append('all the grass.')
                self.sen_instIdx_map.append(grass_ids)
                self.sorted_indices.extend(grass_ids)

        if len(road_ids) >= 1:
            texts.append('the road.')
            self.sen_instIdx_map.append(road_ids)
            self.sorted_indices.extend(road_ids)

        return " ".join(texts)

    def get_text(self):
        """
            1. get text of weather summary
            2. get text of single objects in the sky
            3. get text of objects on the ground
            4. get text of road and grass
        """
        self.sorted_indices = []
        self.sen_instIdx_map = []
        # weather_summary = self.get_weather_summary()
        weather_singles = self.get_weather_singles()
        ground_items = self.get_ground_items()
        grass_road_text = self.get_grass_road_text()

        texts = [weather_singles, ground_items, grass_road_text]
        return " ".join([text for text in texts if text != ""]), self.sorted_indices, self.sen_instIdx_map


def png2text(pred_boxes, pred_class_ids, dataset_base_dir):
    items = data_generation_util.init_all_items(pred_boxes, pred_class_ids, dataset_base_dir)
    # print("items (in the order of item.oid)", [item.oid for item in items])
    solution = ImageToText(items)
    full_caption, sorted_indices_list, sen_instIdx_map_list = solution.get_text()
    return full_caption, sorted_indices_list, sen_instIdx_map_list
