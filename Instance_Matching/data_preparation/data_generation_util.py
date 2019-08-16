# -*- coding: utf-8 -*-
import json
import math
import queue
import os
import scipy.io

import numpy as np

IMAGE_LENGTH = IMAGE_WIDTH = IMAGE_HEIGHT = 768

NEAR_DISTANCE = 200

TYPES = ["unmovable", "tree", "movable"]
CATEGORIES_UNMOVABLE = ["house", "bus", "truck", "car", "bench", "chair"]
CATEGORIES_TREE = ["tree"]
CATEGORIES_MOVABLE = ["people", "horse", "cow", "sheep", "pig", "cat", "dog", "chicken", "duck", "rabbit", "bird",
                      "butterfly"]

# 16 valid categories
INSTANCE = CATEGORIES_UNMOVABLE + CATEGORIES_TREE + CATEGORIES_MOVABLE + \
           ["cloud", "sun", "moon", "star"] + \
           ["road", "grass"] + ["others"]

RANK = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
        "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third", "twenty-fourth",
        "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth", "thirtieth"]

NUMBER = [" ", "two", "three", "four", "five", "six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
          "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
          "Twenty-one", "Twenty-two", "Twenty-three", "Twenty-four", "Twenty-five", "Twenty-six",
          "Twenty-seven", "Twenty-eight", "Twenty-nine", "Thirty"]

RELATIVE_DIRECTIONS = ["left front", "front", "right front", "right", "left",
                       "left back", "back", "right back"]

DIRECTIONS = ["on the left front of", "in front of", "on the right front of", "on the right of", "on",
              "under", "on the left of", "on the left back of", "behind", "on the right back of"]
PSEUDO_DIRECTIONS = ["around", "among"]

HORIZONAL_DIRECTIONS = ["leftmost", "left second", "middle", "right second", "rightmost"]
VERTICAL_DIRECTIONS = ["topmost", "top second", "middle", "bottom second", "bottommost"]


def get_type(category):
    if category in CATEGORIES_UNMOVABLE:
        return "unmovable"
    elif category in CATEGORIES_TREE:
        return "tree"
    elif category in CATEGORIES_MOVABLE:
        return "movable"
    else:
        return "other"


def get_opposite_relative_direction(relative_direction):
    index = len(RELATIVE_DIRECTIONS) - 1 - RELATIVE_DIRECTIONS.index(relative_direction)
    opposite_relative_direction = RELATIVE_DIRECTIONS[index]
    return opposite_relative_direction


def get_opposite_direction(direction):
    if direction in DIRECTIONS:
        index = len(DIRECTIONS) - 1 - DIRECTIONS.index(direction)  # ???????
        return DIRECTIONS[index]
    elif direction in PSEUDO_DIRECTIONS:
        index = 1 - PSEUDO_DIRECTIONS.index(direction)
        return PSEUDO_DIRECTIONS[index]
    else:
        raise Exception("Undefined direction ", direction)


class ItemOrGroupTypeError(TypeError):
    def __init__(self, o):
        TypeError.__init__(self, "item_or_group must be type of Item or ItemGroup. got %s" % type(o))


class Position(object):
    """
    a 2D point of @code{Item}

    Attributes:
        left: distance to the left of the image
        top: distance to the top of the image

    """

    def __init__(self, left, top):
        super(Position, self).__init__()
        self.left = left
        self.top = top

    def showPosition(self):
        print("position(left, top) is (%d,%d)" % (self.left, self.top))

    # determine the direction to another position
    def is_above(self, position):
        return self.top < position.top

    def is_above_equal(self, position):
        return self.top <= position.top

    def is_below(self, position):
        return self.top > position.top

    def is_below_equal(self, position):
        return self.top >= position.top

    def is_left_of(self, position):
        return self.left < position.left

    def is_left_equal(self, position):
        return self.left <= position.left

    def is_right_of(self, position):
        return self.left > position.left

    def is_right_equal(self, position):
        return self.left >= position.left

    def is_right_below_equal(self, position):
        return self.is_right_equal(position) and self.is_below_equal(position)

    def is_left_above_equal(self, position):
        return self.is_left_equal(position) and self.is_above_equal(position)

    def is_vertical_to(self, position):
        return self.left == position.left

    def is_horizontal_to(self, position):
        return self.top == position.top

    def is_coincide_with(self, position):
        return self.left == position.left and self.top == position.top

    def get_degree_to(self, position):
        """
            return the degree of the angle formed by the horizontal line and the line connected to two position

        Returns:
            When the angle is Obtuse Angle, we use its complementary Acute Angle.
            When dx == 0, the angle degree is 90
        """
        dx = abs(self.left - position.left)
        dy = abs(self.top - position.top)
        if dx == 0:
            return 90
        return math.atan(dy / dx) / math.pi * 180


class Size(object):
    """
    the size of @code{Item}

    """

    def __init__(self, width, height):
        super(Size, self).__init__()
        self.width = width
        self.height = height

    def showSize(self):
        print("width is", self.width, "and height is", self.height)


class Item(object):
    """
    every item has its many attributes
    """

    def __init__(self, category, oid, position, size, id):
        """
        Args:
            category: e.g. "tree", "car"
            type: see TYPES
            oid: e.g. "tree3", "cat2"
            position?the 2D position
            size?height and width
            id: the idx in the input instance list
        """
        super(Item, self).__init__()
        self.category = category
        self.type = get_type(self.category)
        self.oid = oid
        self.position = position
        self.size = size
        self.id = id
        self.center = Position(position.left + size.width / 2, position.top + size.height / 2)
        self.bottom_center = Position(position.left + size.width / 2, position.top + size.height)
        self.is_grouped = False
        self.num = 1
        self._name = None
        self._reference = None
        self.direction = None

    def showPosition(self):
        self.position.showPosition()

    def showSize(self):
        self.size.showSize()

    def showCenter(self):
        print("center(left, top) is (%d, %d)" % (self.center.left, self.center.top))

    def showSelf(self):
        print("--------------an %s----------------" % self.category)
        print("rect size:")
        self.size.showSize()
        print("-------------------------------------")

    def get_degree_to(self, item):
        """
            return the degree of the angle formed by the horizontal line and the line connected to two bottom centers
        """
        return self.bottom_center.get_degree_to(item.bottom_center)

    def is_horizontal_to(self, item_or_group):
        if isinstance(item_or_group, Item):
            item = item_or_group
            return self.get_degree_to(item) <= 30
        elif isinstance(item_or_group, ItemGroup):
            item_group = item_or_group
            return item_or_group.top <= self.bottom <= item_group.bottom
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_vertical_to(self, item_or_group):
        if isinstance(item_or_group, Item):
            item = item_or_group
            return self.get_degree_to(item) >= 60
        elif isinstance(item_or_group, ItemGroup):
            item_group = item_or_group
            return self.is_center_horizontally_inside(item_group)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_bottom_edge_below(self, item_or_group):
        """
            determine if its own bottom edge is lower than @code{item_or_group}'s bottom edge
        """
        if isinstance(item_or_group, Item):
            item = item_or_group
            return self.bottom > item.bottom
        elif isinstance(item_or_group, ItemGroup):
            item_group = item_or_group
            return self.bottom > item_group.bottom
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_bottom_edge_above(self, item_or_group):
        """
            determine if its own bottom edge is higher than @code{item_or_group}'s bottom edge
        """
        return not self.is_bottom_edge_below(item_or_group)

    def is_bottom_edge_below_top_foot(self, item_or_group):
        item_group = item_or_group
        return self.bottom > item_group.top

    def is_foot_vertically_inside(self, item_or_group):
        """
            determine if its own bottom edge is within than @code{item_or_group}'s bottom edge and top edge
        """
        if isinstance(item_or_group, ItemGroup):
            return not self.is_bottom_edge_below(item_or_group) and self.is_bottom_edge_below_top_foot(item_or_group)
        else:
            raise TypeError("item_or_group must be instance of ItemGroup.")

    @property
    def left(self):
        return self.position.left

    @property
    def right(self):
        return self.position.left + self.size.width

    @property
    def top(self):
        return self.position.top

    @property
    def bottom(self):
        return self.position.top + self.size.height

    @property
    def width(self):
        return self.size.width

    @property
    def height(self):
        return self.size.height

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        if not isinstance(reference, (Item, ItemGroup)):
            raise TypeError("reference must be type of Item or ItemGroup. got %s" % type(reference))
        self._reference = reference

    @property
    def be_verb(self):
        return "is"

    @property
    def image_position(self):
        room_length = IMAGE_LENGTH / 3.0
        image_positions = ["left", "middle", "right"]
        y = self.center.left / room_length
        if y <= 1.25:
            y = 0
        elif y >= 1.75:
            y = 2
        else:
            y = 1
        if image_positions[y] == "middle":
            return "in the " + image_positions[y]
        else:
            return "on the " + image_positions[y]

    def is_near(self, item):
        edge_dist = self.edge_distance(item)
        # print(edge_dist)
        return edge_dist <= NEAR_DISTANCE

    # the 5 functions below: compare the center pos with others
    def is_center_right_of(self, item_or_group):
        if isinstance(item_or_group, Item) or isinstance(item_or_group, ItemGroup):
            return self.center.is_right_of(item_or_group.center)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_center_left_of(self, item_or_group):
        if isinstance(item_or_group, Item) or isinstance(item_or_group, ItemGroup):
            return not self.is_center_right_of(item_or_group)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_center_above(self, item_or_group):
        if isinstance(item_or_group, Item) or isinstance(item_or_group, ItemGroup):
            return self.center.is_above(item_or_group.center)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_center_below(self, item_or_group):
        if isinstance(item_or_group, Item) or isinstance(item_or_group, ItemGroup):
            return not self.center.is_above(item_or_group.center)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_center_horizontally_inside(self, item_or_group):
        """
            determine if its horizontal pos is within than @code{item_or_group}'s left edge and right edge
        """
        if isinstance(item_or_group, (Item, ItemGroup)):
            return item_or_group.left <= self.center.left <= item_or_group.right
        else:
            raise ItemOrGroupTypeError(item_or_group)

    def edge_distance(self, item_or_group):
        """
        edge_distance means: for vertical distance @code{dy} between A and B, and horizontal distance @code{dx},
            edge_dist = sqrt(dx ^ 2 + dy ^ 2)

        vertical distance means the smallest gap between A's bbox and B's bbox vertically
        horizontal distance means the smallest gap between As bbox and B's bbox horizontally

        Args:
            dy: >=0
            dx: >=0
            item_or_group: @code{Item} or @code{ItemGroup}
        """
        if not isinstance(item_or_group, (Item, ItemGroup)):
            raise ItemOrGroupTypeError(item_or_group)
        dy = 0
        item_or_group_real_top = item_or_group.top if isinstance(item_or_group, Item) else item_or_group.real_top
        if self.top > item_or_group.bottom:
            dy = abs(self.top - item_or_group.bottom)
        elif self.bottom < item_or_group_real_top:
            dy = abs(self.bottom - item_or_group_real_top)

        dx = 0
        if self.left > item_or_group.right:
            dx = abs(self.left - item_or_group.right)
        elif self.right < item_or_group.left:
            dx = abs(self.right - item_or_group.left)

        return math.sqrt(dx ** 2 + dy ** 2)

    def get_single_noun(self):
        if self.category == "people":
            return "person"
        return self.category

    def get_noun(self):
        return self.get_single_noun()

    def get_noun_with_num(self, is_sentence_head=False, more_nature=False):
        if more_nature:
            return "%s" % self.get_single_noun(), 'a', self.get_single_noun()

        if is_sentence_head:
            return "%s" % self.get_single_noun(), 'a', self.get_single_noun()
        else:
            return "%s" % self.get_single_noun(), 'a', self.get_single_noun()

    def get_position_to_item(self, item=None):
        """
        use to compare two item with the same category
        """
        degree = self.get_degree_to(item)
        if self.is_bottom_edge_above(item):  # behind the item
            if degree > 65:
                return "back"
            elif 30 <= degree <= 65:
                if self.is_center_right_of(item):
                    return "right back"
                else:
                    return "left back"
        elif self.is_bottom_edge_below(item):
            if degree > 65:
                return "front"
            elif 30 <= degree <= 65:
                if self.is_center_right_of(item):
                    return "right front"
                else:
                    return "left front"
        if self.is_center_right_of(item):
            return "right"
        elif self.is_center_left_of(item):
            return "left"

    def get_position_to_item_group(self, item_group=None):
        """
            use to compare a item and a itemgroup with the same category
        """
        if self.is_vertical_to(item_group):
            if self.is_bottom_edge_above(item_group):
                return "back"
            else:
                return "front"
        elif self.is_horizontal_to(item_group):
            if self.is_center_right_of(item_group):
                return "right"
            else:
                return "left"
        elif self.is_center_right_of(item_group):
            if self.is_bottom_edge_above(item_group):
                return "right back"
            else:
                return "right front"
        elif self.is_center_left_of(item_group):
            if self.is_bottom_edge_above(item_group):
                return "left back"
            else:
                return "left front"

    def get_position_to(self, reference):
        """
            item --position--> self
            use to compare objects in same category but different group

        """
        if isinstance(reference, Item):
            return self.get_position_to_item(reference)
        elif isinstance(reference, ItemGroup):
            return self.get_position_to_item_group(reference)
        else:
            raise TypeError("reference must be instance of Item or ItemGroup")

    def get_name(self, is_sentence_head=False):
        if not is_sentence_head:
            return self._name
        else:
            return "the" + self._name[3:]

    def set_name(self, num_total, reference=None, index=None, opposite_direction=None):
        """
            set name for each group. the name is used for being reference
        """
        if num_total < 1:
            raise ValueError("num_total must be greater than 1.")

        if opposite_direction is not None:
            direction = get_opposite_relative_direction(opposite_direction)
            self._name = "the %s %s" % (direction, self.get_single_noun())
            return None

        if num_total == 1:
            self._name = "the %s" % self.get_single_noun()
            return None
        elif num_total == 2:
            direction = self.get_position_to(reference)
            self._name = "the %s %s" % (direction, self.get_single_noun())
            return direction
        else:
            self._name = "the left %s" % (self.get_single_noun())
            return None


class ItemGroup(object):
    def __init__(self, items: [Item]):
        if not isinstance(items, type([Item])):
            raise TypeError("type of items must be [Item]")
        if len(items) == 0:
            raise ValueError("length of items must be longer than 0.")
        self.items = items
        self.category = items[0].category
        self.type = get_type(self.category)
        self._avg_item = self.get_avg_item(items)

        # the left-most, right-most, top-most, bottom-most of the groups of Items
        self.right = max([item.right for item in items])
        self.left = min([item.left for item in items])
        self.real_top = min([item.top for item in items])  # the real top
        self.top = min([item.bottom for item in items])  # the top-most of foot
        self.bottom = max([item.bottom for item in items])  # the bottom-most of foot
        self.bottom_center = Position((self.left + self.right) / 2, self.bottom)

        self._name = None
        self._reference = None
        self.direction = None

    @staticmethod
    def get_avg_item(items):
        """
            count the avg pos and size of the grouped items, and return a avg Item
        """
        category = items[0].category
        oid = items[0].oid
        total_left = 0
        total_top = 0

        total_height = 0
        total_width = 0
        num = 0

        for item in items:
            total_left += item.position.left
            total_top += item.position.top
            total_height += item.size.height
            total_width += item.size.width
            num += 1

        position = Position(total_left / num, total_top / num)
        size = Size(total_width / num, total_height / num)
        id = -1

        item = Item(category, oid, position, size, id)
        item.num = num
        return item

    @staticmethod
    def index_trans(old_total_count, new_total_count):
        assert 0 < new_total_count and new_total_count <= old_total_count
        if new_total_count == 1:
            return [0]
        elif new_total_count == 2:
            return [0, 4]
        elif new_total_count == 3:
            return [0, 2, 4]
        elif new_total_count == 4:
            return [0, 1, 3, 4]
        elif new_total_count == 5:
            return [0, 1, 2, 3, 4]

    @property
    def num(self):
        return len(self.items)

    @property
    def center(self):
        return self._avg_item.center

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        if not isinstance(reference, (Item, ItemGroup)):
            raise TypeError("reference must be type of Item or ItemGroup. got %s" % type(reference))
        self._reference = reference

    @property
    def be_verb(self):
        return "are"

    @property
    def noun(self):
        return self.get_plural_noun()

    @property
    def image_position(self):
        room_length = IMAGE_LENGTH / 3.0
        image_positions = ["left", "middle", "right"]
        y = self.center.left / room_length
        if y <= 1.25:
            y = 0
        elif y >= 1.75:
            y = 2
        else:
            y = 1
        if image_positions[y] == "middle":
            return "in the " + image_positions[y]
        else:
            return "on the " + image_positions[y]

    def get_noun_with_num(self, is_sentence_head=False, more_nature=False):
        count_str = NUMBER[self.num - 1] if self.num < 6 else ''

        if more_nature:
            count_str = count_str.lower()
            return "Another %s %s" % (count_str, self.get_plural_noun())

        return "%s %s" % (count_str, self.get_plural_noun()), count_str, self.get_plural_noun()

    def is_center_right_of(self, item_or_group):
        if isinstance(item_or_group, Item) or isinstance(item_or_group, ItemGroup):
            return self.center.is_right_of(item_or_group.center)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_center_left_of(self, item_or_group):
        if isinstance(item_or_group, Item) or isinstance(item_or_group, ItemGroup):
            return not item_or_group.is_center_right_of(self)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_center_above(self, item_or_group):
        if isinstance(item_or_group, Item) or isinstance(item_or_group, ItemGroup):
            return self.center.is_above(item_or_group.center)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_bottom_edge_above(self, item_group):
        if isinstance(item_group, ItemGroup):
            return self.bottom < item_group.bottom
        else:
            raise TypeError("item_group must be instance of ItemGroup.")

    def is_center_horizontally_cover(self, item_or_group):
        if isinstance(item_or_group, (Item, ItemGroup)):
            return self.left <= item_or_group.center.left <= self.right
        else:
            raise ItemOrGroupTypeError(item_or_group)

    def is_center_horizontally_inside(self, item_or_group):
        if isinstance(item_or_group, (Item, ItemGroup)):
            return item_or_group.left <= self.center.left <= item_or_group.right
        else:
            raise ItemOrGroupTypeError(item_or_group)

    def is_foot_vertically_cover(self, item_group):
        return self.top <= item_group.bottom <= self.bottom

    def is_foot_vertically_inside(self, item_group):
        return item_group.top <= self.bottom <= item_group.bottom

    def is_around_another_item_or_group(self, item_or_group):
        # e.g.: item_or_group: house; self: tree
        if isinstance(item_or_group, (Item, ItemGroup)):
            return self.is_center_horizontally_cover(item_or_group) and self.is_foot_vertically_cover(item_or_group)
        else:
            raise ItemOrGroupTypeError(item_or_group)

    def is_among_another_group(self, item_group):
        # item_group: e.g. trees; self: e.g. cats
        if not isinstance(item_group, ItemGroup):
            raise TypeError("item_group must be of type ItemGroup. got %s" % type(item_group))

        return self.is_center_horizontally_inside(item_group) and self.is_foot_vertically_inside(item_group)

    def get_degree_to(self, item_or_group):
        """
            return the degree of the angle formed by the horizontal line and the line connected to two center

        """
        return self.center.get_degree_to(item_or_group.center)

    def is_vertical_to(self, item_or_group):
        if isinstance(item_or_group, Item):
            item = item_or_group
            return self.get_degree_to(item) >= 60
        elif isinstance(item_or_group, ItemGroup):
            item_group = item_or_group
            return self.is_center_horizontally_cover(item_group) or self.is_center_horizontally_inside(item_group)
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def is_horizontal_to(self, item_or_group):
        if isinstance(item_or_group, Item):
            item = item_or_group
            return self.get_degree_to(item) <= 30
        elif isinstance(item_or_group, ItemGroup):
            item_group = item_or_group
            return item_or_group.top <= self.center.top <= item_group.bottom
        else:
            raise TypeError("item_or_group must be instance of Item or ItemGroup.")

    def get_plural_noun(self):
        if self.category in ["people", "sheep"]:
            return self.category
        if self.category[-1] == "y":
            return self.category[:-1] + "ies"
        elif self.category[-1] == "s" or self.category[-2:] == "ch":
            return self.category + "es"
        else:
            return self.category + "s"

    def get_noun(self):
        return self.get_plural_noun()

    def get_position_to_item_group(self, item_group):
        """
        :param item_group: another group with the same category
        :return: the direction to rhe item_group
        """
        if self.is_vertical_to(item_group):
            if self.is_bottom_edge_above(item_group):
                return "back"
            else:
                return "front"
        if self.is_center_right_of(item_group):
            return "right"
        else:
            return "left"

    def get_position_to(self, item_or_group):
        if isinstance(item_or_group, Item):
            opposite_direction = item_or_group.get_position_to(self)
            return get_opposite_relative_direction(opposite_direction)
        elif isinstance(item_or_group, ItemGroup):
            return self.get_position_to_item_group(item_or_group)
        else:
            raise TypeError("item_or_group must be type of Item or ItemGroup.")

    def edge_distance(self, item_or_group):
        if not isinstance(item_or_group, (Item, ItemGroup)):
            raise ItemOrGroupTypeError(item_or_group)
        dy = 0
        item_or_group_real_top = item_or_group.top if isinstance(item_or_group, Item) else item_or_group.real_top
        if self.top > item_or_group.bottom:
            dy = abs(self.top - item_or_group.bottom)
        elif self.bottom < item_or_group_real_top:
            dy = abs(self.bottom - item_or_group_real_top)

        dx = 0
        if self.left > item_or_group.right:
            dx = abs(self.left - item_or_group.right)
        elif self.right < item_or_group.left:
            dx = abs(self.right - item_or_group.left)

        return math.sqrt(dx ** 2 + dy ** 2)

    def get_name(self, is_sentence_head=False):
        if not is_sentence_head:
            return self._name
        else:
            return "the" + self._name[3:]

    def set_name(self, num_total, reference=None, index=None, opposite_direction=None):
        if num_total < 1:
            raise ValueError("num_total must be greater than 1.")

        if opposite_direction is not None:
            direction = get_opposite_relative_direction(opposite_direction)
            self._name = "the %s %s" % (direction, self.get_plural_noun())
            return None

        if num_total == 1:
            self._name = "the %s" % self.get_plural_noun()
            return None
        elif num_total == 2:
            direction = self.get_position_to(reference)
            self._name = "the %s %s" % (direction, self.get_plural_noun())
            return direction
        else:
            self._name = "the left %s" % (self.get_plural_noun())
            return None

    def sort_items_by_pos_left(self):
        # for item in self.items:
        #     print('item.left', item.left)
        self.items = sorted(self.items, key=lambda x: x.left)
        # for item in self.items:
        #     print('item.left', item.left)

    def sort_items_by_pos_bottom(self):
        self.items = sorted(self.items, key=lambda x: x.bottom)

    def check_items_distribution_horizontal(self):
        group_width = self.right - self.left
        group_height = self.bottom - self.real_top
        # print('group_width', group_width)
        # print('group_height', group_height)
        items = [item for item in self.items]
        items_sorted = sorted(items, key=lambda x: x.bottom)
        accu_bottom_gap = 0
        for i in range(1, len(items_sorted)):
            accu_bottom_gap += items_sorted[i].bottom - items_sorted[i - 1].bottom

        return not accu_bottom_gap > 0.5 * group_height

    def find_reference(self, reference_candidate=None, other_group_in_same_big_category=None):
        ## if more than 6 items, no need to describe each single item
        if self.category == 'house' and len(self.items) < 6:
            self.sort_items_by_pos_left()
            for index, item in enumerate(self.items):
                item.direction = HORIZONAL_DIRECTIONS[
                    self.index_trans(len(HORIZONAL_DIRECTIONS), len(self.items))[index]]
            if len(self.items) == 2:
                simply_direc = ['left', 'right']
                for index, item in enumerate(self.items):
                    item.direction = simply_direc[index]
        elif self.category in (CATEGORIES_UNMOVABLE + CATEGORIES_TREE + CATEGORIES_MOVABLE) and len(self.items) < 6:
            assert len(self.items) > 1
            if len(self.items) == 2:
                # regard the first item in the group as the ref
                self.sort_items_by_pos_left()
                item_left = self.items[0]
                item_left.direction = 'left'
                item_right = self.items[1]
                item_right.direction = 'right'

                # item.reference = self.items[0]
                # item.direction = ItemCollection.get_dir_of_item(self.items[0], item)
            else:
                # from left to right or up to down
                if self.check_items_distribution_horizontal():
                    self.sort_items_by_pos_left()
                    for index, item in enumerate(self.items):
                        item.direction = HORIZONAL_DIRECTIONS[
                            self.index_trans(len(HORIZONAL_DIRECTIONS), len(self.items))[index]]
                else:
                    self.sort_items_by_pos_bottom()
                    for index, item in enumerate(self.items):
                        item.direction = VERTICAL_DIRECTIONS[
                            self.index_trans(len(VERTICAL_DIRECTIONS), len(self.items))[index]]

                # TODO: search for 'near'
                if self.category in CATEGORIES_MOVABLE:
                    self.near_found = False
                    for index, item in enumerate(self.items):
                        other_items = [single_item for tmp_idx, single_item in enumerate(self.items) if
                                       tmp_idx != index]
                        # 1. search UNMOVEABLE and Tree
                        if reference_candidate is not None:
                            for unmove_or_tree in reference_candidate:
                                if isinstance(unmove_or_tree, (Item)):
                                    others_min_distance = min(
                                        [unmove_or_tree.edge_distance(item_) for item_ in other_items])
                                    self_distance = unmove_or_tree.edge_distance(item)
                                    if others_min_distance - self_distance >= 50 and self_distance <= 50:
                                        item.reference = unmove_or_tree
                                        self.near_found = True
                                        break
                        if self.near_found:
                            break

                        # 2. search MOVEABLE
                        if other_group_in_same_big_category is not None:
                            for other_group in other_group_in_same_big_category:
                                if isinstance(other_group, (Item)):
                                    others_min_distance = min(
                                        [other_group.edge_distance(item_) for item_ in other_items])
                                    self_distance = other_group.edge_distance(item)
                                    # print('other_group', other_group.category, 'item', item.category)
                                    # print('22 others_min_distance', others_min_distance)
                                    # print('22 self_distance', self_distance)
                                    if others_min_distance - self_distance >= 50 and self_distance <= 50:
                                        item.reference = other_group
                                        self.near_found = True
                                        break
                        if self.near_found:
                            break

                    # print('self.near_found', self.near_found)


class ItemCollection(object):
    MODES = ["unmovable", "tree", "movable"]

    def __init__(self, dict_collection, unmovable_reference=None, tree_reference=None):
        """

        :param dict_collection: e.g. {'bus': [Util2.Item], 'house': [Util2.Item]} /
                                     {'tree': [Util2.ItemGroup, Util2.ItemGroup]} /
                                     {'people': [Util2.ItemGroup]}
        :param unmovable_reference: unmovable objects as references
        :param tree_reference: tree objects as references

        """
        self.dict_collection = dict_collection
        # print(dict_collection)
        if len(dict_collection) == 0:
            self.collection = []
            return

        if not isinstance(dict_collection, type({str: []})):
            raise TypeError("type of dict_collection must be {str: []}.", type(dict_collection))

        # set name for each Item or ItemGroup in dict_collection
        self.num_total = {}
        for category in dict_collection:
            items_group = dict_collection[category]  # list, e.g. [Util2.Item] / [Util2.ItemGroup, Util2.ItemGroup]
            ItemCollection._set_name_for_item_or_groups(items_group)
            num_total = 0
            for item in items_group:
                num_total += item.num
            self.num_total[category] = num_total

        ## all of the 3 big category are sorted by category first and then pos_left
        self.collection = ItemCollection.sort_dict_by_category_and_pos_left(dict_collection)

        if unmovable_reference is None and tree_reference is None:
            self.mode = "unmovable"
            self.find_reference()
        elif tree_reference is None:
            self.mode = "tree"
            self.unmovable = ItemCollection.sort_dict_by_category_and_pos_left(unmovable_reference)
            self.find_reference(self.unmovable)
        else:
            self.mode = "movable"
            self.unmovable = ItemCollection.sort_dict_by_category_and_pos_left(unmovable_reference)
            self.trees = ItemCollection.sort_dict_by_category_and_pos_left(tree_reference)
            self.find_reference(self.unmovable, self.trees)

    def find_reference(self, unmovable_reference=None, tree_reference=None):
        """
            find reference for each Item or ItemGroup

            # TODO: For convinence, the search for the reference for a group when no reference_candidate currently,
            # TODO: which means such group should look for reference within the same big category, is simplied by
            # TODO: using the former group as reference. This is not natural in some unusual cases,
            # TODO: e.g., only three groups of dogs in the image without other objects

        """
        if self.mode == "unmovable":
            # unmovable group objects should be referenced by unmovable group objects
            # the first object doesn't need reference, only described by "in the middle", "in the right" etc.
            for index, cur_item in enumerate(self.collection):
                # if cur_item is a group. find the reference of the group
                reference_candidate = self.collection[:index]
                if index != 0:
                    nearest = min(reference_candidate, key=lambda x: cur_item.edge_distance(x))
                    cur_item.direction = ItemCollection.get_dir_of(nearest, cur_item)
                    cur_item.reference = nearest

                # foreach items in a group, find their own reference
                if isinstance(cur_item, ItemGroup):
                    cur_item.find_reference()

        elif self.mode == "tree":
            reference_candidate = unmovable_reference
            if len(reference_candidate) == 0:
                # no unmovable objects as ref, other tree should find tree as ref
                for index, cur_item in enumerate(self.collection):
                    reference_candidate_temp = self.collection[:index]
                    if index != 0:
                        nearest = min(reference_candidate_temp, key=lambda x: cur_item.edge_distance(x))
                        cur_item.direction = ItemCollection.get_dir_of(nearest, cur_item)
                        cur_item.reference = nearest

                    if isinstance(cur_item, ItemGroup) and len(cur_item.items) > 1:
                        cur_item.find_reference()
            else:
                # with unmovable group objects as ref
                for index, cur_item in enumerate(self.collection):
                    nearest = min(reference_candidate, key=lambda x: cur_item.edge_distance(x))
                    cur_item.direction = ItemCollection.get_dir_of(nearest, cur_item)
                    cur_item.reference = nearest

                    if isinstance(cur_item, ItemGroup) and len(cur_item.items) > 1:
                        cur_item.find_reference()

        elif self.mode == "movable":
            # reference should be chosen from the nearest from [unmovable_reference + tree_reference]
            reference_candidate = unmovable_reference + tree_reference  # list of group

            if len(reference_candidate) == 0:
                # no unmovable/tree objects as reference_candidate:
                for index, cur_item in enumerate(self.collection):
                    if index >= 1:
                        former_item = self.collection[index - 1]
                        cur_item.direction = ItemCollection.get_dir_of(former_item, cur_item)
                        cur_item.reference = former_item

                    if isinstance(cur_item, ItemGroup):
                        cur_item.find_reference()
            else:
                ## with reference_candidate
                for index, cur_item in enumerate(self.collection):
                    nearest = min(reference_candidate, key=lambda x: cur_item.edge_distance(x))
                    cur_item.direction = ItemCollection.get_dir_of(nearest, cur_item)
                    cur_item.reference = nearest

                    # foreach items in a group, find their own reference
                    if isinstance(cur_item, ItemGroup):
                        other_group_in_same_big_category \
                            = [single_item for tmp_idx, single_item in enumerate(self.collection) if tmp_idx != index]
                        cur_item.find_reference(reference_candidate, other_group_in_same_big_category)
        else:
            raise ValueError("mode must be one of the %s. got %s" % (self.MODES, self.mode))

    @staticmethod
    def get_dir_of_item(this, that: Item):
        """
            get the direction of 'that' to 'this' ('that'    'this')
            e.g. if 'that' is on the left of 'this', return 'on the left of'

            :param that: type of Item
        """
        # this: e.g. trees; that: e.g. cat (inner the trees)
        if this.category == 'tree' and isinstance(this, ItemGroup):
            # if this is trees group, should have the attr of 'among'
            # a single tree cannot 'around' anything, so no 'around' here
            if that.is_center_horizontally_inside(this) and that.is_foot_vertically_inside(this):
                return "among"

        if that.is_vertical_to(this):
            if that.is_bottom_edge_above(this):
                return "behind"
            else:
                return "in front of"
        elif that.is_horizontal_to(this):
            if that.is_center_right_of(this):
                return "on the right of"
            else:
                return "on the left of"
        elif that.is_center_right_of(this):
            if that.is_bottom_edge_above(this):
                return "on the right back of"
            else:
                return "on the right front of"
        elif that.is_center_left_of(this):
            if that.is_bottom_edge_above(this):
                return "on the left back of"
            else:
                return "on the left front of"

    @staticmethod
    def get_dir_of_item_group(this, that: ItemGroup):
        """
        :param this: Item/ItemGroup
        :param that: ItemGroup
        """
        ## this: Item, e.g. car; that: ItemGroup, e.g. tree
        if isinstance(this, Item):
            opposite_direction = ItemCollection.get_dir_of_item(that, this)
            return get_opposite_direction(opposite_direction)

        elif not isinstance(this, ItemGroup):
            raise TypeError("this must be of type Item or ItemGroup. got %s" % type(this))

        ## this: ItemGroup, e.g. house; that: ItemGroup, e.g. tree
        if that.category == 'tree' and that.is_around_another_item_or_group(this):
            return "around"
        ## this: ItemGroup, e.g. tree; that: ItemGroup, e.g. cat
        elif this.category == 'tree' and that.is_among_another_group(this):
            return "among"

        if that.is_vertical_to(this):
            if that.is_bottom_edge_above(this):
                return "behind"
            else:
                return "in front of"
        if that.is_center_right_of(this):
            return "on the right of"
        else:
            return "on the left of"

    @staticmethod
    def get_dir_of(this, that):
        if isinstance(that, Item):
            return ItemCollection.get_dir_of_item(this, that)
        elif isinstance(that, ItemGroup):
            return ItemCollection.get_dir_of_item_group(this, that)
        else:
            raise ItemOrGroupTypeError(that)

    @staticmethod
    def sort_dict_by_category_and_pos_left(dict_collection: {str: []}):
        """
        foreach category in order of CATEGORIES_UNMOVABLE, sort the items or groups by left position
        """
        list_sorted_by_cate_and_pos = []
        for kind in (CATEGORIES_UNMOVABLE + CATEGORIES_TREE + CATEGORIES_MOVABLE):
            if dict_collection.get(kind) is not None:
                list_sorted_by_pos = []
                for item_or_group in dict_collection[kind]:
                    list_sorted_by_pos.append(item_or_group)
                list_sorted_by_cate_and_pos += sorted(list_sorted_by_pos, key=lambda x: x.left)

        return list_sorted_by_cate_and_pos

    @staticmethod
    def get_collections(ground_items):
        """
        :param ground_items: a list of instance category in order

        Args:
            unmovable: e.g. {'bus': [<Util2.Item>], 'house': [<Util2.Item>]}
            trees: e.g. {'tree': [<Util2.ItemGroup>, <Util2.ItemGroup>]}
            movable: e.g. {'people': [<Util2.ItemGroup>]}
        """
        unmovable = ItemCollection._get_collection(ground_items, CATEGORIES_UNMOVABLE)
        trees = ItemCollection._get_collection(ground_items, CATEGORIES_TREE)
        movable = ItemCollection._get_collection(ground_items, CATEGORIES_MOVABLE)

        return ItemCollection(unmovable), ItemCollection(trees, unmovable), ItemCollection(movable, unmovable, trees)

    @staticmethod
    def _get_collection(ground_items, categories):
        items = [item for item in ground_items if item.category in categories]
        collection = ItemCollection._merge_same_item(items)
        return collection

    @staticmethod
    def _merge_same_item(items: [Item]):
        """
        :param items: are in the same big category, e.g. unmoveable objects
        :return:
        """
        if len(items) == 0:
            return {}
        items_set = set(items)
        process_queue = queue.Queue()
        items_groups = {}

        while len(items_set) > 0:
            # let the first item be processed.
            first_item_in_group = items_set.pop()
            process_queue.put(first_item_in_group)
            group_category = first_item_in_group.category
            items_group = [first_item_in_group]

            while not process_queue.empty():
                cur_item = process_queue.get()
                for item in items_set:
                    if item.category == group_category and item.is_near(
                            cur_item) and not item.is_grouped:  # item is in the same group
                        process_queue.put(item)
                        items_group.append(item)
                        item.is_grouped = True
                    items_set = items_set.difference(items_group)  # update the items_set

            if not items_groups.get(group_category):
                items_groups[group_category] = []

            if len(items_group) > 1:
                items_groups[group_category].append(ItemGroup(items_group))
            else:
                items_groups[group_category].append(items_group[0])

        return items_groups

    @staticmethod
    def _set_name_for_item_or_groups(item_or_groups: []):
        """
        ????????items????item????
        :param item_or_groups: list with elemants Item/ItemGroup
        :return:
        """
        num_total = len(item_or_groups)
        for item in item_or_groups:
            if not isinstance(item, (Item, ItemGroup)):
                raise TypeError("type of each entry in items must be Item or ItemGroup")
        if num_total == 1:  # only one
            item_or_groups[0].set_name(num_total=1)
        elif num_total == 2:  # only two
            first = item_or_groups[0]
            second = item_or_groups[1]
            direction = first.set_name(num_total=2, reference=second)
            second.set_name(num_total=2, opposite_direction=direction)
        else:
            item_or_groups = sorted(item_or_groups, key=lambda x: x.bottom)
            for index, item_or_group in enumerate(item_or_groups):
                item_or_group.set_name(num_total=num_total, index=index)

    @staticmethod
    def get_single_noun(category):
        if category == "people":
            return "person"
        return category

    @staticmethod
    def get_plural_noun(category):
        if category in ["people", "sheep"]:
            return category
        if category[-1] == "y":
            return category[:-1] + "ies"
        elif category[-1] == "s" or category[-2:] == "ch":
            return category + "es"
        else:
            return category + "s"

    @staticmethod
    def get_noun(category, num):
        if num == 1:
            return ItemCollection.get_single_noun(category)
        elif num > 1:
            return ItemCollection.get_plural_noun(category)
        else:
            raise ValueError("num must be larger than 0. got %d" % num)

    def get_description(self):
        sorted_indices_list = []
        sen_instIdx_map_list = []
        each_description = []

        if len(self.collection) == 0:
            return {"des": "", "sorted_indices": [], "sen_instIdx_map": []}

        # if there are >= 2 groups
        for cate_name in self.dict_collection:
            cate_list = self.dict_collection[cate_name]
            # print('####', cate_name, cate_list)
            if len(cate_list) > 1:
                all_inst_ids = []
                for item_or_group_ in cate_list:
                    if isinstance(item_or_group_, Item):
                        all_inst_ids.append(item_or_group_.id)
                    if isinstance(item_or_group_, ItemGroup):
                        for item_ in item_or_group_.items:
                            all_inst_ids.append(item_.id)

                noun_with_num, num_phrase, noun_phrase = cate_list[0].get_noun_with_num(is_sentence_head=True)
                assert len(all_inst_ids) >= 2
                if len(all_inst_ids) == 2:
                    each_description.append(" both the %s." % (noun_phrase))
                    sen_instIdx_map_list.append(all_inst_ids)
                else:
                    each_description.append(" all the %s." % (noun_phrase))
                    sen_instIdx_map_list.append(all_inst_ids)

        ## describe each Item or Group
        for coll_idx, item_or_group in enumerate(self.collection):
            ## 1. first the summary of a Group
            noun_with_num, num_phrase, noun_phrase = item_or_group.get_noun_with_num(is_sentence_head=True)
            be_verb = item_or_group.be_verb

            description = ''

            if item_or_group.reference is not None:
                # change 'a' to 'another' when dict_collection.len == 2
                # item_or_group_cate = item_or_group.category
                # if len(self.dict_collection[item_or_group_cate]) == 2 \
                #         and self.dict_collection[item_or_group_cate].index(item_or_group) > 0:
                #     noun_with_num = item_or_group.get_noun_with_num(is_sentence_head=True, more_nature=True)

                if item_or_group.category == 'tree' and item_or_group.reference.category == 'tree':
                    dir_refined = item_or_group.direction
                    if 'of' in dir_refined:
                        dir_refined = dir_refined[:-3]
                    if 'behind' in dir_refined:
                        dir_refined = 'on ' + dir_refined
                    description += "the %s %s." % (noun_with_num, dir_refined)
                elif item_or_group.category in ['bird', 'butterfly']:
                    # special cases
                    description += "the %s %s %s." % (noun_with_num, "near", item_or_group.reference.get_name())
                else:
                    # describe the relation with the reference
                    description += "the %s %s %s." % (noun_with_num,
                                                      item_or_group.direction, item_or_group.reference.get_name())

                ## for Item, add id directly
                if isinstance(item_or_group, Item):
                    sen_instIdx_map_list.append([item_or_group.id])
                if isinstance(item_or_group, ItemGroup):
                    sen_instIdx_map_list.append([item.id for item in item_or_group.items])

            # describe the position directly if without reference
            image_position = item_or_group.image_position
            description += " the %s %s." % (noun_with_num, image_position)
            if isinstance(item_or_group, Item):
                sorted_indices_list.extend([item_or_group.id])
                sen_instIdx_map_list.append([item_or_group.id])
            if isinstance(item_or_group, ItemGroup):
                ## add instIdx to the list
                sorted_indices_list.extend([item.id for item in item_or_group.items])
                sen_instIdx_map_list.append([item.id for item in item_or_group.items])

            ## -----------------------------------------------------------
            ## No reference and direction
            # for single object, can have 'the bus'
            if isinstance(item_or_group, Item):
                if len(self.dict_collection[item_or_group.category]) == 1:
                    description += " the %s." % (noun_with_num)
                    sen_instIdx_map_list.append([item_or_group.id])

            # for the only group, can have 'the trees' / 'all the trees'
            if isinstance(item_or_group, ItemGroup):
                if len(self.dict_collection[item_or_group.category]) == 1:
                    if num_phrase == 'two':
                        description += " both the %s." % (noun_with_num)
                        sen_instIdx_map_list.append([item.id for item in item_or_group.items])

                        description += " both the %s." % (noun_phrase)
                        sen_instIdx_map_list.append([item.id for item in item_or_group.items])

                    else:
                        description += " all the %s." % (noun_with_num)
                        sen_instIdx_map_list.append([item.id for item in item_or_group.items])

                        description += " all the %s." % (noun_phrase)
                        sen_instIdx_map_list.append([item.id for item in item_or_group.items])

                    description += " the %s." % (noun_with_num)
                    sen_instIdx_map_list.append([item.id for item in item_or_group.items])

                    description += " the %s." % (noun_phrase)
                    sen_instIdx_map_list.append([item.id for item in item_or_group.items])

            ## -----------------------------------------------------------

            if isinstance(item_or_group, ItemGroup):
                if len(item_or_group.items) < 6:
                    ## 2. every item in the group should be described

                    # only >=2 group need to find a group_direction_ref
                    group_direction_ref_str = ''
                    if item_or_group.reference is not None and len(self.dict_collection[item_or_group.category]) > 1:
                        group_direction_ref_str = '%s %s' % (
                        item_or_group.direction, item_or_group.reference.get_name())

                    for index, item in enumerate(item_or_group.items):
                        item_index = item.id
                        item_noun = item.get_noun()
                        direction = item.direction

                        sen_instIdx_map_list.append([item_index])

                        ## the first Item in group might have no reference
                        ## all house and [UNMOVABLE + MOVABLE](>2 items) have no reference but direction
                        if item.category == 'house' or \
                                item.category in (CATEGORIES_UNMOVABLE + CATEGORIES_TREE + CATEGORIES_MOVABLE) \
                                and len(item_or_group.items) > 2:
                            # for len(item_or_group.items) == 3~5

                            if item.reference is not None:
                                description += " the %s near %s." % (item_noun, item.reference.get_name())
                            else:
                                if group_direction_ref_str != '':
                                    if direction in (HORIZONAL_DIRECTIONS + VERTICAL_DIRECTIONS + ['left', 'right']):
                                        description += " the %s %s %s." % (
                                        direction, item_noun, group_direction_ref_str)
                                else:
                                    if len(self.dict_collection[item_or_group.category]) == 1:
                                        if 'second' not in direction:
                                            description += " the %s %s the %s." % \
                                                           (item_noun, 'in' if direction == 'middle' else 'on',
                                                            direction)

                                            description += " the %s %s." % \
                                                           (direction, item_noun)
                                            sen_instIdx_map_list.append([item_index])
                                        else:
                                            description += " the second %s on the %s." % \
                                                           (item_noun, direction[:direction.find('second') - 1])
                                    else:
                                        sen_instIdx_map_list = sen_instIdx_map_list[:-1]
                        else:
                            # for len(item_or_group.items) == 2
                            if item.reference is not None:
                                reference_name = item.reference.get_name()
                                description += " the %s %s %s." % (item_noun, direction, reference_name)
                            else:
                                if group_direction_ref_str != '':
                                    description += " the %s %s %s." % (direction, item_noun, group_direction_ref_str)
                                else:
                                    if len(self.dict_collection[item_or_group.category]) == 1:
                                        if 'of' in direction:
                                            direction = direction[:-3]
                                            description += " the %s %s." % (item_noun, direction)
                                        elif 'behind' in direction:
                                            direction = 'on ' + direction
                                            description += " the %s %s." % (item_noun, direction)
                                        elif 'left' in direction or 'right' in direction:
                                            description += " the %s on the %s." % (item_noun, direction)

                                            description += " the %s %s." % (direction, item_noun)
                                            sen_instIdx_map_list.append([item_index])
                                    else:
                                        sen_instIdx_map_list = sen_instIdx_map_list[:-1]

                            # if index == 0:
                            #     if item.reference is not None:
                            #         reference_name = item.reference.get_name()
                            #         description += " the %s %s %s." % (item_noun, direction, reference_name)
                            #     else:
                            #         if group_direction_ref_str != '':
                            #             if 'left' in item.image_position:
                            #                 description += " the left %s %s." % (item_noun, group_direction_ref_str)
                            #             elif 'middle' in item.image_position:
                            #                 description += " the middle %s %s." % (item_noun, group_direction_ref_str)
                            #             elif 'right' in item.image_position:
                            #                 description += " the right %s %s." % (item_noun, group_direction_ref_str)
                            #         else:
                            #             if len(self.dict_collection[item_or_group.category]) == 1:
                            #                 description += " the %s %s." % (item_noun, item.image_position)
                            #             else:
                            #                 sorted_indices_list = sorted_indices_list[:-1]
                            #                 sen_instIdx_map_list = sen_instIdx_map_list[:-1]
                            # elif index > 0:
                            #     if len(self.dict_collection[item_or_group.category]) == 1:
                            #         if 'of' in direction:
                            #             direction = direction[:-3]
                            #         if 'behind' in direction:
                            #             direction = 'on ' + direction
                            #         description += " the %s %s." % (item_noun, direction)
                            #     else:
                            #         sorted_indices_list = sorted_indices_list[:-1]
                            #         sen_instIdx_map_list = sen_instIdx_map_list[:-1]

            each_description.append(description)

        description = []
        description.extend(each_description)
        des = " ".join(description)
        return {"des": des, "sorted_indices": sorted_indices_list, "sen_instIdx_map": sen_instIdx_map_list}


def init_categories_map(dataset_base_dir):
    categories_map_ = {}
    color_map_mat_path = os.path.join(dataset_base_dir, 'colorMapC46.mat')
    colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
    for i in range(len(colorMap)):
        cat_name = colorMap[i][0][0]
        categories_map_[i + 1] = cat_name

    return categories_map_


def init_all_items(pred_boxes, pred_class_ids, dataset_basedir):
    categories_map = init_categories_map(dataset_basedir)

    items = []
    for i in range(0, len(pred_class_ids)):
        category = categories_map[pred_class_ids[i]]
        if category in INSTANCE:  # if the category is valid
            left = pred_boxes[i][1]
            top = pred_boxes[i][0]
            width = pred_boxes[i][3] - pred_boxes[i][1]
            height = pred_boxes[i][2] - pred_boxes[i][0]
            id = i
            oid = "%s%d" % (category, i)
            position = Position(left, top)
            size = Size(width, height)
            item = Item(category, oid, position, size, id)
            items.append(item)
    return items
