from .space import Space
import numpy as np
import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
import torch
import random

class PackingContinuous(gym.Env):
    def __init__(self,
                 setting,
                 container_size=(10, 10, 10),
                 item_set=None, data_name=None, load_test_data=False,
                 internal_node_holder=80, leaf_node_holder=50, next_holder=1, shuffle=False,
                 sample_from_distribution=True,
                 sample_left_bound=0.1,
                 sample_right_bound=0.5,
                 **kwags):

        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = next_holder
        self.shuffle = shuffle
        self.bin_size = container_size

        if sample_from_distribution:
            self.size_minimum = sample_left_bound
            self.sample_left_bound = sample_left_bound
            self.sample_right_bound = sample_right_bound
        else:
            self.size_minimum = np.min(np.array(item_set))

        self.setting = setting
        self.item_set = item_set
        self.sample_from_distribution = sample_from_distribution

        self.orientation = 6 if self.setting == 2 else 2
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder)

        if not load_test_data:
            assert item_set is not None
            self.box_creator = RandomBoxCreator(item_set)
        if load_test_data:
            self.box_creator = LoadBoxCreator(data_name)

        self.test = load_test_data
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=self.space.height,
            shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,)
        )
        self.next_box_vec = np.zeros((self.next_holder, 9))
        self.LNES = 'EMS'

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]

    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (
            self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])

    def reset(self, **kwargs):
        self.box_creator.reset()
        self.packed = []
        self.space.reset()
        self.box_creator.generate_box_size()
        return self.cur_observation()

    def cur_observation(self):
        boxes = []
        leaf_nodes = []
        self.next_box = self.gen_next_box()

        if self.test:
            self.next_den = self.next_box[3] if self.setting == 3 else 1
            self.next_box = [round(self.next_box[0], 3), round(self.next_box[1], 3), round(self.next_box[2], 3)]
        else:
            if self.setting < 3:
                self.next_den = 1
            else:
                self.next_den = np.random.random()
                while self.next_den == 0:
                    self.next_den = np.random.random()

        boxes.append(self.space.box_vec)
        leaf_nodes.append(self.get_possible_position())

        next_box = sorted(list(self.next_box))
        self.next_box_vec[:, 3:6] = next_box
        self.next_box_vec[:, 0] = self.next_den
        self.next_box_vec[:, -1] = 1

        return np.reshape(np.concatenate((*boxes, *leaf_nodes, self.next_box_vec)), (-1,))

    def gen_next_box(self):
        if self.sample_from_distribution and not self.test:
            if self.setting == 2:
                return (
                    round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                    round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                    round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3)
                )
            else:
                return (
                    round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                    round(np.random.uniform(self.sample_left_bound, self.sample_right_bound), 3),
                    np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                )
        return self.box_creator.preview(1)[0]

    def get_possible_position(self):
        if self.LNES == 'EMS':
            allPostion = self.space.EMSPoint(self.next_box, self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box, self.setting)
        else:
            raise ValueError('Invalid LNES option')

        if self.shuffle:
            np.random.shuffle(allPostion)

        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []

        for position in allPostion:
            xs, ys, zs, xe, ye, ze = position
            x = xe - xs
            y = ye - ys
            z = ze - zs

            if self.space.drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                if len(tmp_list) >= self.leaf_node_holder:
                    break

        if tmp_list:
            leaf_node_vec[:len(tmp_list)] = np.array(tmp_list)

        return leaf_node_vec

    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0:
            return (0, 0, 0), self.next_box
        x = round(leaf_node[3] - leaf_node[0], 6)
        y = round(leaf_node[4] - leaf_node[1], 6)
        record = [0, 1, 2]
        for r in record:
            if abs(x - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        for r in record:
            if abs(y - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        z = self.next_box[record[0]]
        return (0, leaf_node[0], leaf_node[1]), (x, y, z)

    def step(self, action):
        if len(action) != 3:
            action, next_box = self.LeafNode2Action(action)
        else:
            next_box = self.next_box

        idx = [round(action[1], 6), round(action[2], 6)]
        rotation_flag = action[0]
        succeeded = self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting)

        if not succeeded:
            reward = 0.0
            terminated = True
            truncated = False
            info = {'counter': len(self.space.boxes), 'ratio': self.space.get_ratio(),
                    'reward': self.space.get_ratio() * 10}
            return self.cur_observation(), reward, terminated, truncated, info

        packed_box = self.space.boxes[-1]

        if self.LNES == 'EMS':
            self.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                               round(packed_box.lx + packed_box.x, 6),
                               round(packed_box.ly + packed_box.y, 6),
                               round(packed_box.lz + packed_box.z, 6)])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, 0])

        box_ratio = self.get_box_ratio()
        self.box_creator.drop_box()
        self.box_creator.generate_box_size()
        reward = box_ratio * 10

        terminated = False
        truncated = False
        info = {'counter': len(self.space.boxes)}
        return self.cur_observation(), reward, terminated, truncated, info
