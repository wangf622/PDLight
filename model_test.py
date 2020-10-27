import json
import os
import pickle
from config import DIC_AGENTS, DIC_ENVS
from copy import deepcopy


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def downsample(path_to_log, i):
    path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
    with open(path_to_pkl, "rb") as f_logging_data:
        logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    os.remove(path_to_pkl)
    with open(path_to_pkl, "wb") as f_subset:
        pickle.dump(subset_data, f_subset)

def downsample_for_system(path_to_log,dic_traffic_env_conf):
    for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
        downsample(path_to_log,i)



# TODO test on multiple intersections
def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf, if_gui):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d"%cnt_round
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)

    if os.path.exists(os.path.join(records_dir, "sumo_env.conf")):
        with open(os.path.join(records_dir, "sumo_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    elif os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    dic_traffic_env_conf["IF_GUI"] = if_gui

    # dump dic_exp_conf
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)


    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    agents = []


    try:
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                                                               path_to_work_directory=dic_path[
                                                                   "PATH_TO_WORK_DIRECTORY"],
                                                               dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = dic_exp_conf["MODEL_NAME"]
            if agent_name=='CoLight_Signal':
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    cnt_round=1,  # useless
                    inter_info=env.list_intersection,
                    intersection_id=str(i)
                )
            else:
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    cnt_round=1,  # useless
                    intersection_id=str(i)
                )
            agents.append(agent)
            

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            if dic_traffic_env_conf["ONE_MODEL"]:
                agents[i].load_network("{0}".format(model_round))
            else:
                agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))


        step_num = 0

        attention_dict = {}
        import time
        START_TIME = time.time()
        phase_time_list = []
        pressure_2_phasetime = {5:11,6:12,7:12,8:13,9:14,10:14,11:15,
                                12:16,13:16,14:17,15:18,16:18,17:19,18:20,19:20}

        while not done and step_num < dic_exp_conf["RUN_COUNTS"]:
            action_list = []

            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):

                if "CoLight" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state
                    action_list, attention = agents[i].choose_action(step_num, one_state)
                    cur_time = env.get_current_time()
                    attention_dict[cur_time] = attention
                elif "GCN" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state
                    # print('one_state:',one_state)
                    action_list = agents[i].choose_action(step_num, one_state)
                    # print('action_list:',action_list)
                elif "SimpleDQNOne" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state
                    if True:
                        action_list = agents[i].choose_action(step_num, one_state)
                    else:
                        action_list = agents[i].choose_action_separate(step_num, one_state)
                else:
                    one_state = state[i]
                    action = agents[i].choose_action(step_num, one_state)
                    action_list.append(action)
            
            pressure = env._get_pressure()
            p_2 = [0]*dic_traffic_env_conf['NUM_INTERSECTIONS']
            for inter_i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
                p_2[inter_i] = pressure[inter_i][action_list[inter_i]]
                
            ave_pressure = sum(p_2)/len(p_2)

            if ave_pressure<5:
                phase_time = 10
            elif ave_pressure>=20:
                phase_time = 20
            else:
                phase_time = pressure_2_phasetime[int(ave_pressure)]
            
            phase_time_list.append(phase_time)            

            next_state, reward, done, _ = env.step(action_list,phase_time)

            state = next_state
            step_num += phase_time
            print('test -',cnt_round,round(ave_pressure,2),phase_time,step_num,time.time()-START_TIME)
        # print('bulk_log_multi_process')
        with open(path_to_log+'/phase_time.txt','w') as phase_time_file:
            phase_time_file.write('\n'.join([str(phase_time) for phase_time in phase_time_list]))
        env.bulk_log_multi_process()
        env.log_attention(attention_dict)

        env.end_sumo()
        if not dic_exp_conf["DEBUG"]:
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                       model_round)
            # print("downsample", path_to_log)
            downsample_for_system(path_to_log, dic_traffic_env_conf)
            # print("end down")

    except:
        error_dir = model_dir.replace("model", "errors")
        if os.path.exists(error_dir):
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("round_%d fail to test model"%cnt_round)
            f.close()
        else:
            os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("round_%d fail to test model"%cnt_round)
            f.close()
