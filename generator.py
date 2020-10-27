import os
import copy
from config import DIC_AGENTS, DIC_ENVS
import time
import sys
from multiprocessing import Process, Pool

class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, best_round=None):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']

        if self.dic_exp_conf["PRETRAIN"]:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
        else:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)  

        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                              path_to_log = self.path_to_log,
                              path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf = self.dic_traffic_env_conf)        
        self.env.reset()

        # every generator's output
        # generator for pretraining
        # Todo pretrain with intersection_id
        if self.dic_exp_conf["PRETRAIN"]:

            self.agent_name = self.dic_exp_conf["PRETRAIN_MODEL_NAME"]
            self.agent = DIC_AGENTS[self.agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round,
                best_round=best_round,
            )

        else:

            start_time = time.time()

            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                agent_name = self.dic_exp_conf["MODEL_NAME"]
                #the CoLight_Signal needs to know the lane adj in advance, from environment's intersection list
                if agent_name=='CoLight_Signal':
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round, 
                        best_round=best_round,
                        inter_info=self.env.list_intersection,
                        intersection_id=str(i)
                    )      
                else:              
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round, 
                        best_round=best_round,
                        intersection_id=str(i)
                    )
                self.agents[i] = agent
            print("Create intersection agent time: ", time.time()-start_time)







    def generate(self):

        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time

        running_start_time = time.time()
        phase_time_list = []
        pressure_2_phasetime = {5:11,6:12,7:12,8:13,9:14,10:14,11:15,
                                12:16,13:16,14:17,15:18,16:18,17:19,18:20,19:20}

        while not done and step_num < self.dic_exp_conf["RUN_COUNTS"]:
            action_list = []
            step_start_time = time.time()

            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):

                if self.dic_exp_conf["MODEL_NAME"] in ["CoLight","GCN", "SimpleDQNOne"]:
                    one_state = state
                    if self.dic_exp_conf["MODEL_NAME"] == 'CoLight':
                        action, _ = self.agents[i].choose_action(step_num, one_state)
                    elif self.dic_exp_conf["MODEL_NAME"] == 'GCN':
                        action = self.agents[i].choose_action(step_num, one_state)
                    else: # simpleDQNOne
                        if True:
                            action = self.agents[i].choose_action(step_num, one_state)
                        else:
                            action = self.agents[i].choose_action_separate(step_num, one_state)
                    action_list = action
                else:
                    one_state = state[i]
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            pressure = self.env._get_pressure()
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

            next_state, reward, done, _ = self.env.step(action_list,phase_time)

            print('gen  -',self.cnt_gen,round(ave_pressure,2),phase_time,self.env.get_current_time()-self.dic_traffic_env_conf["MIN_ACTION_TIME"], time.time()-step_start_time)
            state = next_state
            step_num += phase_time
        running_time = time.time() - running_start_time
        with open(self.path_to_log+'/phase_time.txt','w') as phase_time_file:
            phase_time_file.write('\n'.join([str(phase_time) for phase_time in phase_time_list]))

        log_start_time = time.time()
        print("start logging")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time

        self.env.end_sumo()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)
