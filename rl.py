import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import trange
import copy
import numpy as np
import csv
import arcpy
from arcpy import env
from arcpy.sa import KernelDensity
import rasterio
from rasterio.windows import Window
from joblib import load
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.impute import KNNImputer
import os
import geopandas as gpd
from rasterio.mask import mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.warp import reproject, Resampling


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN:
    def __init__(self, n_states, n_actions):
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.n_actions = n_actions
        self.n_states = n_states
        self.batch_size = 32
        self.gamma = 0.9
        self.learn_step_counter = 0  # target网络学习计数
        self.memory_counter = 0  # 记忆计数
        self.memory_space = MEMORY_CAPACITY
        self.memory = np.zeros((self.memory_space, n_states * 2 + 2))
        self.cost = []  # 记录损失值

    def choose_action(self, x, epsilon):
        legal_act_pos = np.argwhere(x != 0)
        legal_act = np.array([i[0] * grid_col + i[1] for i in legal_act_pos])
        mask = np.array([1 if i in legal_act else 0 for i in range(self.n_actions)])
        if np.random.uniform() < epsilon:
            state = torch.unsqueeze(torch.FloatTensor(x.flatten()), 0)
            action_value = self.eval_net.forward(state)
            action_mask_value = action_value[0].data.numpy() * mask
            action = np.argmax(action_mask_value)
        else:
            action = np.random.choice(legal_act)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_space
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()))
        self.learn_step_counter += 1

        # 使用记忆库中批量数据
        sample_index = np.random.choice(self.memory_space, self.batch_size)
        memory = self.memory[sample_index, :]  # 抽取的记忆单元，并逐个提取
        state = torch.FloatTensor(memory[:, :self.n_states])
        action = torch.LongTensor(memory[:, self.n_states:self.n_states + 1])
        reward = torch.LongTensor(memory[:, self.n_states + 1:self.n_states + 2])
        next_state = torch.FloatTensor(memory[:, -self.n_states:])

        mask_batch = np.zeros_like(next_state)
        next_state_batch_grid = next_state.reshape(-1, grid_row, grid_col).numpy()
        for i in range(next_state_batch_grid.shape[0]):
            legal_act_pos = np.argwhere(next_state_batch_grid[i] != 0)
            if len(legal_act_pos) > 0:
                legal_act = np.array([i[0] * grid_col + i[1] for i in legal_act_pos])
                mask_batch[i] = np.array([1 if i in legal_act else 0 for i in range(self.n_actions)])
        mask_batch_torch = torch.from_numpy(mask_batch)
        q_eval = self.eval_net(state).gather(1, action)
        q_next = self.target_net(next_state).detach() * mask_batch_torch
        q_target = reward + self.gamma * q_next.max(1)[0].unsqueeze(1)
        loss = self.loss(q_eval, q_target)
        self.cost.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path, train_model):
        self.eval_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
        if train_model:
            self.eval_net.train()
            self.target_net.train()
        else:
            self.eval_net.eval()


def get_args():
    with rasterio.open(feature_to_raster_path['Temperature']) as src:
        transform = src.transform
        lbrt = src.bounds
        nodata_value = src.nodata
        meta = src.meta.copy()
        meta.update(count=1, dtype='float64', compress='lzw')
    return transform, lbrt, nodata_value, meta


def csv2arr():
    csv_file = os.path.join(root_path, 'use_file/ori_1.csv')
    f = open(csv_file, newline='')
    arr0 = np.array(list(csv.reader(f))[1:])[:, :2].astype(float)  # 维度，精度
    ent_num = arr0.shape[0]  # 企业数量
    arr1 = np.zeros((ent_num, 5))
    for i in range(ent_num):
        arr1[i, :2] = arr0[i, :2]
        arr1[i, 4] = i

        arr1[i, 2] = (84 - arr1[i, 0]) // 2
        arr1[i, 3] = (arr1[i, 1] + 180) // 2
    grid = np.zeros((70, 180))
    for item in arr1:
        x, y = item[2:4].astype(int)
        grid[x, y] += 1
    return arr1, grid


def get_tif11_full_feature():
    feature_arr = np.zeros((11, map_row, map_col))
    i = 0
    for key, key_path in feature_to_raster_path.items():
        with rasterio.open(key_path) as src:
            d = src.read(1)
            feature_arr[i,:d.shape[0],:d.shape[1]] = d[:map_row, :map_col]
        i += 1
    full_feature = np.zeros((grid_row, grid_col, 6400, 11))
    feature_arr[np.isnan(feature_arr)] = -100
    feature_arr[feature_arr < -100] = -100

    imputer = KNNImputer(n_neighbors=5, missing_values=-100)
    for i in trange(grid_row, desc='knn'):
        for j in range(grid_col):
            feature_grid = feature_arr[:, window_size * i:window_size * (i + 1), window_size * j:window_size * (j + 1)]
            feature = feature_grid.reshape(11, -1)
            imputed_data = imputer.fit_transform(feature.T)
            if imputed_data.shape[1] == 11:
                full_feature[i, j] = imputed_data
    np.save(os.path.join(root_path, 'use_file/knn.npy'), full_feature)
    return full_feature


def get_ori_result_arr():
    spatial_ref = arcpy.SpatialReference(4326)
    csv_path = os.path.join(root_path, f'use_file/ori_1.csv')
    shp_path = os.path.join(root_path, f'temp_file/points_layer_-1')

    arcpy.management.XYTableToPoint(csv_path, shp_path, "Longitude", "Latitude", None, spatial_ref)
    cell_size = 0.025
    search_radius = 200000
    area_unit = "SQUARE_KILOMETERS"
    out_density = KernelDensity(shp_path, None, cell_size, search_radius, area_unit, "DENSITIES", "GEODESIC")
    data = np.zeros((map_row, map_col))
    data_out = arcpy.RasterToNumPyArray(out_density)
    data[:data_out.shape[0],:data_out.shape[1]]=data_out
    ori_result_arr = np.zeros((map_row, map_col))
    if os.path.exists(os.path.join(root_path, 'use_file/catboost_ori.npy')):
        return np.load(os.path.join(root_path, 'use_file/catboost_ori.npy'))

    for i in trange(grid_row, desc='catboost'):
        for j in range(grid_col):
            t1 = data[window_size * i:window_size * (i + 1), window_size * j:window_size * (j + 1)].reshape(-1, 1)
            tif11_data = full_11feature[i, j]  # 57600*11
            if np.max(tif11_data) != np.min(tif11_data):
                tif12_data = np.hstack((tif11_data[:, :1], t1, tif11_data[:, 1:]))  # 把可变特征插入到11特征的第二列形成12特征
                tif12_data_pd = pd.DataFrame(tif12_data, columns=columns_ordered)
                window_scaled = scaler.transform(tif12_data_pd)
                preds = model.predict_proba(window_scaled)[:, 1].reshape(window_size, window_size)
                ori_result_arr[window_size * i:window_size * (i + 1), window_size * j:window_size * (j + 1)] = preds
    np.save(os.path.join(root_path, 'use_file/catboost_ori.npy'), ori_result_arr)
    return ori_result_arr



def get_new_p(idx, i,now_eps):
    st_time = time.time()
    csv1_file = os.path.join(root_path, f'temp_file/csv_{now_eps}_{i}.csv')
    shp_file = os.path.join(root_path, f'temp_file/points_layer_{i}')
    spatial_ref = arcpy.SpatialReference(4326)
    arcpy.management.XYTableToPoint(csv1_file, shp_file, "Longitude", "Latitude", None, spatial_ref)

    cell_size = 0.025
    search_radius = 200000
    area_unit = "SQUARE_KILOMETERS"
    out_density = KernelDensity(shp_file, None, cell_size, search_radius, area_unit, "DENSITIES", "GEODESIC")
    data = np.zeros((map_row, map_col))
    data_out = arcpy.RasterToNumPyArray(out_density)
    data[:data_out.shape[0],:data_out.shape[1]]=data_out
    st_row = np.clip(idx[0] - 1, 0, grid_row)
    ed_row = np.clip(idx[0] + 2, 0, grid_row)
    st_col = np.clip(idx[1] - 1, 0, grid_col)
    ed_col = np.clip(idx[1] + 2, 0, grid_col)
    t1_time = time.time()
    # print(f't1  {t1_time-st_time}')
    for i in range(st_row, ed_row):
        for j in range(st_col, ed_col):
            tif1_data = data[window_size * i:window_size * (i + 1), window_size * j:window_size * (j + 1)].reshape(-1,
                                                                                                                   1)
            tif11_data = full_11feature[i, j]  # 57600*11
            if np.max(tif11_data) != np.min(tif11_data):
                tif12_data = np.hstack((tif11_data[:, :1], tif1_data, tif11_data[:, 1:]))  # 把可变特征插入到11特征的第二列形成12特征
                tif12_data_pd = pd.DataFrame(tif12_data, columns=columns_ordered)
                window_scaled = scaler.transform(tif12_data_pd)
                preds = model.predict_proba(window_scaled)[:, 1].reshape(window_size, window_size)
                ori_result_arr[window_size * i:window_size * (i + 1), window_size * j:window_size * (j + 1)] = preds
    t2_time=time.time()
    # print(f't2  {t2_time-t1_time}')
    p = tif2_p()
    # print(f't3  {time.time() - t2_time}')
    return p


def tif2_p():
    a = ori_result_arr
    m = (a > 0) & (b > 0) & (a * b < 1e9)
    cn = np.multiply(a, b, where=(a >= 0.25) & m)
    cv = cn[cn >= 0]
    p = cv.sum()
    return p


def draw(i, lst):
    plt.figure()
    plt.plot(lst)
    plt.savefig(os.path.join(root_path, f'png/{i}.png'))
    plt.close()


def delete_non_csv_files(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        for file in folder.glob('*'):
            if file.is_file() and file.suffix != '.csv':
                try:
                    file.unlink()
                except:
                    pass


def delete_tempfiles(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        for file in folder.glob('*'):
            if file.is_file():
                try:
                    file.unlink()
                except:
                    pass


if __name__ == '__main__':
    root_path = r'F:\1126'


    delete_tempfiles(os.path.join(root_path, 'temp_file'))

    bs = os.path.join(root_path, 'use_file/11282020_pop_resampled18_padded11_resampled.tif')
    b = rasterio.open(os.path.join(root_path, bs)).read(1).astype(np.float64)


    columns_ordered = [
        'Temperature', 'densityPyr', 'ocs_030', 'Rainy Precipitation', 'underground level',
        'bbod100200', 'water source.1', 'cfvo100200', 'Dry Precipitation', 'Aridity',
        'Dry Temperature', 'standards'
    ]
    feature_to_raster_path = {
        'Temperature': os.path.join(root_path, 'use_file/final_平均温度掩膜提取.tif'),
        'ocs_030': os.path.join(root_path, 'use_file/final_ocs030.tif'),
        'Rainy Precipitation': os.path.join(root_path, 'use_file/final_最潮湿月份降水量.tif'),
        'underground level': os.path.join(root_path, 'use_file/final_underground_level.tif'),
        'bbod100200': os.path.join(root_path, 'use_file/final_cbod12f.tif'),
        'water source.1': os.path.join(root_path, 'use_file/final_surwater_source.tif'),
        'cfvo100200': os.path.join(root_path, 'use_file/final_cfvo100200f.tif'),
        'Dry Precipitation': os.path.join(root_path, 'use_file/final_最干旱月份的降水量.tif'),
        'Aridity': os.path.join(root_path, 'use_file/final_aridity.tif'),
        'Dry Temperature': os.path.join(root_path, 'use_file/final_旱季气温掩膜提取.tif'),
        'standards': os.path.join(root_path, 'use_file/final_standards.tif')
    }
    transform, lbrt, nodata_value, meta = get_args()

    arcpy.CheckOutExtension("Spatial")
    env.workspace = os.path.join(root_path, 'temp_file')  # 确保路径正确
    env.overwriteOutput = True
    env.extent = f'{lbrt[0]} {lbrt[1]} {lbrt[2]} {lbrt[3]}'

    scaler = load(os.path.join(root_path, 'use_file/28scaler.joblib'))
    model = CatBoostClassifier()
    model.load_model(os.path.join(root_path, 'use_file/28catboost_model.cbm'))
    ori_csv_arr, ori_grid = csv2arr()

    # hyper-parameters
    retrain = True

    MEMORY_CAPACITY = 20000
    max_com = 3283
    grid_row, grid_col = ori_grid.shape
    window_size = 80

    NUM_ACTIONS = grid_row * grid_col
    NUM_STATES = grid_row * grid_col
    map_row, map_col = grid_row * window_size, grid_col * window_size

    dqn = DQN(NUM_STATES, NUM_ACTIONS)
    # (state, action, r, next_state)

    max_eps = 0
    if retrain:
        max_eps = max([int(i[4:-4]) for i in os.listdir(os.path.join(root_path, f'model'))])
        dqn.load_model(os.path.join(root_path, f'model/eps={max_eps}.pth'), 1)
        mem_temp = np.load(os.path.join(root_path, f'model/eps={max_eps}.npy'))
        for i in range(MEMORY_CAPACITY):
            dqn.store_transition(mem_temp[i][:NUM_STATES], mem_temp[i][NUM_STATES],
                                 mem_temp[i][NUM_STATES + 1], mem_temp[i][NUM_STATES + 2:])

    episodes = 400

    if not os.path.exists(os.path.join(root_path, 'use_file/knn.npy')):
        full_11feature = get_tif11_full_feature()  # row*col*57600*11
    else:
        full_11feature = np.load(os.path.join(root_path, 'use_file/knn.npy'))
    ori_result = get_ori_result_arr()
    ori_result_arr = copy.deepcopy(ori_result)
    ori_p = tif2_p()


    act_arr = np.load(os.path.join(root_path, 'ans.npy'))
    act_list = np.argwhere(act_arr[:, 0] > 0).flatten()
    act_grid = np.zeros_like(ori_grid)
    for act in act_list:
        grid_pos = np.array([act // 180, act % 180])
        act_grid[*grid_pos]=ori_grid[*grid_pos]


    for eps in range(max_eps + 1, max_eps + episodes + 1):
        last_p = ori_p
        csv_arr = copy.deepcopy(ori_csv_arr)
        grid = copy.deepcopy(act_grid)
        ori_result_arr = copy.deepcopy(ori_result)
        r_list = []
        for step in trange(max_com, desc=f'eps:{eps}'):
            step_time = time.time()
            state = grid.flatten()
            if not retrain and dqn.memory_counter < MEMORY_CAPACITY:
                legal_act_pos = np.argwhere(grid != 0)
                legal_act = np.array([i[0] * grid_col + i[1] for i in legal_act_pos])
                action = np.random.choice(legal_act)
            else:
                action = dqn.choose_action(grid, 0.9)

            idx = np.array([action // grid_col, action % grid_col])  # 动作->数组的行列序号

            # 动作->新状态,目前随机移除非0格1个企业
            np.random.shuffle(csv_arr)
            arr = csv_arr[(csv_arr[:, 2] == idx[0]) & (csv_arr[:, 3] == idx[1])]
            if arr.shape[0] >= 1:
                id = int(arr[0, 4])  # 移除第一个满足条件的，获得其索引
                row = np.where(csv_arr[:, 4] == id)
                csv_arr = np.delete(csv_arr, row, axis=0)
                grid[idx[0], idx[1]] -= 1
            new_csvdata = [['Latitude', 'Longitude']] + csv_arr[:, :2].tolist()
            csv_name = os.path.join(root_path, f'temp_file/csv_{eps}_{step}.csv')
            f = open(csv_name, 'w', newline='')
            csvwriter = csv.writer(f)
            csvwriter.writerows(new_csvdata)
            f.close()
            p = get_new_p(idx, step, eps)
            r = (last_p - p) / 100000
            last_p = p
            next_state = grid.flatten()
            dqn.store_transition(state, action, r, next_state)

            eps_temp_file = os.path.join(root_path, 'state_file', f'eps{eps}')
            if not os.path.exists(eps_temp_file):
                os.makedirs(eps_temp_file)
            np.savez(os.path.join(eps_temp_file, f'step{step}.npz'), state=state, ar=np.array([action, r]),
                     next_state=next_state)

            r_list.append(r)
            state = next_state

            delete_non_csv_files(os.path.join(root_path, 'temp_file'))
            if retrain or dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
            # print(f'{idx[0]},{idx[1]}由{ori_grid_num}减少为{grid[idx[0], idx[1]]},人口减少{r * 100000},单步耗时{time.time() - step_time}\n')
            if p < ori_p*(-0.00425*min(eps,46)+0.665):
            # if p < ori_p * 0.11:
                print(p)
                break
        draw(eps, r_list)
        mem_data = np.array(dqn.memory)
        np.save(os.path.join(root_path, f'model/eps={eps}.npy'), mem_data)
        dqn.save_model(os.path.join(root_path, f'model/eps={eps}.pth'))
