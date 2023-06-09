import os
import sys
sys.path.append(os.getcwd())

from khrylib.rl.envs.common import mujoco_env
from gym import spaces
from khrylib.utils import *
from khrylib.utils.transformation import quaternion_from_euler
#from khrylib.nn_world.data_collector import DataCollector
from motion_imitation.utils.tools import get_expert
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
import scipy.fftpack as fftpack

mujoco_joints = ['lfemur_z', 'lfemur_y', 'lfemur_x', 'ltibia_x',
                 'lfoot_z', 'lfoot_y', 'lfoot_x', 'rfemur_z', 
                 'rfemur_y', 'rfemur_x', 'rtibia_x', 'rfoot_z', 
                 'rfoot_y', 'rfoot_x', 'upperback_z', 'upperback_y', 
                 'upperback_x', 'lowerneck_z', 'lowerneck_y', 
                 'lowerneck_x', 'lclavicle_z', 'lclavicle_y', 
                 'lhumerus_z', 'lhumerus_y', 'lhumerus_x', 'lradius_x', 
                 'rclavicle_z', 'rclavicle_y', 'rhumerus_z', 'rhumerus_y', 
                 'rhumerus_x', 'rradius_x']

#FOOT_JOINTS = ["lfoot_x", "lfoot_y", "lfoot_z", "rfoot_x", "rfoot_y", "rfoot_z"]
FOOT_JOINTS = []

class HumanoidEnv(mujoco_env.MujocoEnv):

    def __init__(self, cfg,filename=None):
        mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file, 15)
        self.cfg = cfg
        self.set_cam_first = set()
        # env specific
        self.end_reward = 0.0
        self.start_ind = 0
        # Print the mass for each link
        self.body_qposaddr = get_body_qposaddr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.set_model_params()
        self.expert = None
        self.filename = filename
        print(self.model.joint_names)
        print("Number of Joints:", len(self.model.joint_names))
        print("Body Names:", self.model.body_names)
        print("Body Mass:", self.model.body_mass)
        ts = time.time()
        self.setup_joint_mapping()
        self.load_expert()
        print(f"Take {time.time()-ts} to load expert")
        #input("Press Enter to Continue")
        self.set_spaces()
        # if self.filename is not None:
        #     self.data_recorder = DataCollector(32,32,26+self.vf_dim, self.model.opt.timestep)

    def setup_joint_mapping(self):
        self.mj_nonfoot = []
        for i in range(len(mujoco_joints)):
            if mujoco_joints[i] not in FOOT_JOINTS:
                self.mj_nonfoot.append(i)

    def load_expert(self,default_id=0):
        expert_qposes = []
        expert_metas = []
        for expert_traj_file in self.cfg.expert_traj_files:
            expert_qpos, expert_meta = pickle.load(open(expert_traj_file, "rb"))
            expert_qposes.append(expert_qpos)
            expert_metas.append(expert_meta)
        self.experts = []
        for i in range(len(expert_qposes)):
            self.experts.append(get_expert(expert_qposes[i], expert_metas[i], self))
        self.expert = self.experts[default_id]
        self.expert_id = default_id

    def set_model_params(self):
        if self.cfg.action_type == 'torque' and hasattr(self.cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cfg.j_stiff
            self.model.dof_damping[6:] = self.cfg.j_damp

    def set_spaces(self):
        cfg = self.cfg
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        print("Num of Dofs:",self.ndof)
        self.vf_dim = 0
        if cfg.residual_force:
            if cfg.residual_force_mode == 'implicit':
                self.vf_dim = 6
            else:
                if cfg.residual_force_bodies == 'all':
                    self.vf_bodies = self.model.body_names[1:]
                else:
                    self.vf_bodies = cfg.residual_force_bodies
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim = self.body_vf_dim * len(self.vf_bodies)
        self.action_dim = self.ndof + self.vf_dim
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def get_obs(self, idx=0):
        if self.cfg.obs_type == 'full':
            obs = self.get_full_obs(idx=idx)
        return obs

    def get_full_obs(self,idx=0):
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cfg.obs_coord).ravel()
        obs = []
        # pos
        if self.cfg.obs_heading:
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cfg.root_deheading:
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:])
        # vel
        if self.cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == 'full':
            obs.append(qvel)
        # phase
        if self.cfg.obs_phase:
            phase = self.get_phase()
            obs.append(np.array([phase]))
        obs = np.concatenate(obs)
        # reference root trajectory
        # Need to assume the heading has already changed orientation.
        q_e = self.get_expert_attr(attr="qpos",ind=self.get_expert_index(idx))
        root_pos = q_e[:7]
        #root_head = quat_mul_vec(get_heading_q(q_e[3:7]),np.array([1.,0.,0.]))
        obs = np.concatenate([obs,root_pos])
        return obs

    def get_ee_pos(self, transform):
        data = self.data
        ee_name = ['lfoot', 'rfoot', 'lwrist', 'rwrist', 'head']
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            # In heading mode the vector is only
            if transform is not None: # transform is a string, not a transformation matrix
                bone_vec = bone_vec - root_pos # Each joint's relative position to root
                bone_vec = transform_vec(bone_vec, root_q, transform) # Express different bone's location in local frame
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_body_quat(self):
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.model.body_names[1:]:
            if body == 'root' or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_com(self):
        return self.data.subtree_com[0, :].copy()
    
    def get_root_quat(self):
        return self.data.qpos[3:7].copy()

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.model.opt.timestep
        nv = self.model.nv
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(self.model.nv, self.model.nv)
        C = self.data.qfrc_bias.copy()
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(cho_factor(M + K_d*dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), overwrite_b=True, check_finite=False)
        return q_accel.squeeze()

    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.model.opt.timestep
        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale[self.mj_nonfoot]
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_pos = cfg.a_ref[self.mj_nonfoot]
        target_pos = base_pos + ctrl_joint

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        # Dimension of cfg.jkp should equals to actuated joint number.
        k_p[6:] = cfg.jkp[self.mj_nonfoot]
        k_d[6:] = cfg.jkd[self.mj_nonfoot]
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -cfg.jkp[self.mj_nonfoot] * qpos_err[6:] - cfg.jkd[self.mj_nonfoot] * qvel_err[6:]
        return torque

    """ RFC-Explicit """
    def rfc_explicit(self, vf):
        qfrc = np.zeros_like(self.data.qfrc_applied)
        for i, body in enumerate(self.vf_bodies):
            body_id = self.model._body_name2id[body]
            contact_point = vf[i*self.body_vf_dim: i*self.body_vf_dim + 3]
            force = vf[i*self.body_vf_dim + 3: i*self.body_vf_dim + 6] * self.cfg.residual_force_scale
            torque = vf[i*self.body_vf_dim + 6: i*self.body_vf_dim + 9] * self.cfg.residual_force_scale if self.cfg.residual_force_torque else np.zeros(3)
            contact_point = self.pos_body2world(body, contact_point)
            force = self.vec_body2world(body, force)
            torque = self.vec_body2world(body, torque)
            mjf.mj_applyFT(self.model, self.data, force, torque, contact_point, body_id, qfrc)
        self.data.qfrc_applied[:] = qfrc

    """ RFC-Implicit """
    def rfc_implicit(self, vf):
        vf *= self.cfg.residual_force_scale
        hq = get_heading_q(self.data.qpos[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        self.data.qfrc_applied[:vf.shape[0]] = vf * 0.2

    def do_simulation(self, action, n_frames, record_data=False):
        t0 = time.time()
        cfg = self.cfg
        for i in range(n_frames):
            ctrl = action
            if cfg.action_type == 'position':
                torque = self.compute_torque(ctrl)
            elif cfg.action_type == 'torque':
                torque = ctrl * cfg.a_scale
            torque = np.clip(torque, -cfg.torque_lim[self.mj_nonfoot], cfg.torque_lim[self.mj_nonfoot])
            self.data.ctrl[:] = torque * 0.2
            #print(torque * 0.2)

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy()
                if cfg.residual_force_mode == 'implicit':
                    self.rfc_implicit(vf)
                else:
                    self.rfc_explicit(vf)

            prev_qpos = self.data.qpos.copy()
            prev_qvel = self.data.qvel.copy()
            self.sim.step()
            if record_data and self.filename is not None:
                self.data_recorder.record(prev_qpos, 
                                          prev_qvel,
                                          self.data.qpos.copy(),
                                          self.data.qvel.copy(),
                                          action.copy())
        #input("Press Enter to Continue")
        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def step(self, a, record_data=False):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.data.qpos.copy()
        self.prev_qvel = self.data.qvel.copy()
        self.prev_bquat = self.bquat.copy()
        # do simulation
        self.do_simulation(a, self.frame_skip, record_data)
        self.cur_t += 1
        self.bquat = self.get_body_quat()
        self.update_expert()
        # get obs
        head_pos = self.get_body_com('head')
        reward = 1.0
        if cfg.env_term_body == 'head':
            fail = self.expert is not None and head_pos[2] < self.expert['head_height_lb'] - 0.1
        else:
            fail = self.expert is not None and self.data.qpos[2] < self.expert['height_lb'] - 0.1
            fail = (self.get_pos_diff(self.cur_t) >= 0.2) or fail # Add tracking criteria
        cyclic = self.expert['meta']['cyclic']
        end =  (cyclic and self.cur_t >= cfg.env_episode_len) or (not cyclic and self.cur_t + self.start_ind >= self.expert['len'] + cfg.env_expert_trail_steps)
        done = fail or end
        obs = self.get_obs(idx=self.cur_t)
        return obs, reward, done, {'fail': fail, 'end': end}

    def reset_model(self):
        cfg = self.cfg
        self.load_expert(default_id = self.get_expert_id())
        if self.expert is not None:
            ind = 0 if self.cfg.env_start_first else self.np_random.randint(self.expert['len'])
            self.start_ind = ind
            init_pose = self.expert['qpos'][ind, :].copy()
            init_vel = self.expert['qvel'][ind, :].copy()
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)
            self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
            self.update_expert()
        else:
            init_pose = self.data.qpos
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.data.qpos[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def update_expert(self):
        expert = self.expert
        if expert['meta']['cyclic']:
            if self.cur_t == 0: # First time
                expert['cycle_relheading'] = np.array([1, 0, 0, 0])
                expert['cycle_pos'] = expert['init_pos'].copy()
            elif self.get_expert_index(self.cur_t) == 0: # Cycle back
                q = expert["qpos"][-1] # Using heading from last frame of previous cycle.
                expert['cycle_relheading'] = quaternion_multiply(get_heading_q(q[3:7]),
                                                              quaternion_inverse(expert['init_heading']))
                expert['cycle_pos'] = np.concatenate((self.data.qpos[:2], expert['init_pos'][[2]]))


    def get_phase(self):
        ind = self.get_expert_index(self.cur_t)
        return ind / self.expert['len']

    def get_expert_index(self, t):
        return (self.start_ind + t) % self.expert['len'] \
                if self.expert['meta']['cyclic'] else min(self.start_ind + t, self.expert['len'] - 1)

    def get_expert_offset(self, t):
        if self.expert['meta']['cyclic']:
            n = (self.start_ind + t) // self.expert['len']
            offset = self.expert['meta']['cycle_offset'] * n
        else:
            offset = np.zeros(2)
        return offset

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind, :]

    # Consider the most significant component in frequency domain
    def get_goal(self):
        traj = self.get_expert_attr("qpos", np.arange(self.expert["len"])).T
        X = fftpack.fft(traj)
        return self.get_expert_attr("qpos", self.expert['len']-1)# + X.argmax(axis=1)/X.shape[1]
        #return X.argmax(axis=1)/X.shape[1]

    def switch_expert(self, idx=None):
        if idx is not None:
            self.expert = self.experts[idx]
            self.expert_id = idx
        else:
            self.expert_id = (self.expert_id+1)%len(self.experts)
            self.expert = self.experts[self.expert_id]
    
    def get_expert_id(self):
        return self.expert_id
    
    def get_num_expert(self):
        return len(self.experts)

    def save_data(self):
        self.data_recorder.save_data(self.filename)

    def get_pos_diff(self, t):
        t = self.get_expert_index(t)
        e_com = self.get_expert_attr("com", t)
        if self.expert['meta']['cyclic']:
            e_rpos = self.get_expert_attr("qpos", t)[:3]
            cycle_h = self.expert["cycle_relheading"]
            cycle_pos = self.expert["cycle_pos"]
            init_pos = self.expert["init_pos"]
            orig_pos = e_rpos.copy()
            e_rpos = quat_mul_vec(cycle_h, e_rpos - init_pos) + cycle_pos
            e_com = quat_mul_vec(cycle_h, e_com - orig_pos) + e_rpos
        diff = np.linalg.norm(e_com - self.get_com())
        #print("Diff:",diff)
        return diff


