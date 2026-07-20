"""PyBullet race-car environment used by the PPO actor/learner trainer.

The car starts at ``(0.5, 0.5)`` and must drive through the closed S-shaped
track to ``(14.5, 7.5)``.  An observation is a 9-element float32 vector:
``[x, y, vx, vy, heading, steer_left, steer_right, steer_vel_left,
steer_vel_right]``.  The action is one normalized steering value in
``[-1, 1]``; throttle is intentionally fixed to ``0.1``.  A wall collision
terminates an episode with -1, reaching the goal terminates it with +3, and
otherwise the reward is ``1 / (1 + distance_to_goal)``.  Episodes are also
truncated after ``max_steps``.  Manual driving established that a successful
run takes about 1400 simulation steps, so the training/evaluation default is
1500.  This is intentionally a tight efficiency constraint, not an arbitrary
timeout; experiments may override it explicitly but should report the value.
Optional start-position and start-heading noise use Gymnasium's per-environment
seeded RNG; training enables small perturbations while manual play keeps the
nominal deterministic start.

Each :class:`RaceCarEnv` owns one PyBullet client.  All physics calls are
explicitly scoped to that client, so independent instances are safe in
different processes (and in one process when each has its own client).  Do
not create an environment before forking an actor: construct it inside the
actor/worker process (or use the ``spawn`` start method).  TensorBoard writers
are optional and must not be shared between processes; aggregate metrics in
the learner instead. ``DirectRaceCarVectorEnv`` in ``actor_env.py`` is the
actor-local synchronous wrapper used by PPO.
"""

import datetime
import math
import os

try:  # Optional: only needed when recording MP4 episodes.
    import imageio
except ImportError:  # pragma: no cover - depends on the training image
    imageio = None
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


DEFAULT_MAX_STEPS = 1500


class RaceCarEnv(gym.Env):
    
    def __init__(self, writer=None, render=False, fps=20,
                 max_steps=DEFAULT_MAX_STEPS, start_position_noise=0.0,
                 start_heading_noise=0.0):
        super().__init__()
        self.writer = writer

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if start_position_noise < 0 or start_heading_noise < 0:
            raise ValueError("start pose noise must be non-negative")
        self.start_position_noise = float(start_position_noise)
        self.start_heading_noise = float(start_heading_noise)


        # 连接物理引擎
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # 设置搜索路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.fps = fps
        p.setTimeStep(1/self.fps, physicsClientId=self.physicsClient)
        p.setRealTimeSimulation(0, physicsClientId=self.physicsClient)

        # 定义动作空间和状态空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.pi,   np.pi, np.pi, np.inf, np.inf]),
            dtype=np.float32
        )

        # 环境参数
        # 人工实测约 1400 步可到终点；默认 1500 是刻意保留的小余量，要求策略高效。
        self.max_steps = int(max_steps)
        self.current_step = 0 #每一回合里的步数计数器
        self.total_step = 0  #环境运行过程中一直累加的计数器
        self.car = None
        self.walls = []
        self.finish_line = None
        self.start_pos = [0.5, 0.5,0.1]  # 起点位置
        self.finish_pos = [14.5, 7.5, 0.1]  # 终点位置
        self.last_pos = None  # 上一步的位置
        self.recordVedio = False
        self.frames = []
        self.prev_state = None

        self.coins = [] #中途奖励的金币
        # 重置环境
        self.reset()

    def _create_track(self):
        """创建封闭的S型赛道"""
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
        self.walls.clear()
        self.coins.clear()

        # 加载地面和赛车
        p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        self.car = p.loadURDF("racecar/racecar.urdf", self.start_pos,
                              physicsClientId=self.physicsClient)

        self._add_wall(0,0, 2,0)
        self._add_wall(0, 0, 0, 4)
        self._add_wall(2, 0, 2, 2)
        self._add_wall(0,4, 8,4)
        self._add_wall(2, 2, 10, 2)
        self._add_wall(10, 2, 10, 6)
        self._add_wall(10,6, 15,6)

        self._add_wall(8, 4, 8, 8)
        self._add_wall(8, 8, 15, 8)
        self._add_wall(15, 6, 15, 8)

        #赛道中间还有金币，鼓励探索
        self.coins.append([3 ,3])
        self.coins.append([9, 5])
        self.coins.append([11, 7])

    def _add_wall(self, startx, starty, endx, endy):
        """
        在(startx,starty)到(endx,endy)之间创建一堵物理墙
        参数:
            startx, starty: 起点坐标 (x,y)
            endx, endy: 终点坐标 (x,y)
        返回:
            wall_id: 创建的墙体ID
        """
        # 墙体参数
        thickness = 0.3  # 厚度0.3米
        height = 0.5  # 高度0.5米
        mass = 0  # 静态墙体；碰撞时不应被赛车推动

        # 计算墙体中心位置和长度
        center_x = (startx + endx) / 2
        center_y = (starty + endy) / 2
        length = math.sqrt((endx - startx) ** 2 + (endy - starty) ** 2)

        # 计算墙体朝向角度（弧度）
        angle = math.atan2(endy - starty, endx - startx)

        # 创建碰撞形状（长方体）
        wall_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length / 2, thickness / 2, height / 2],
            physicsClientId=self.physicsClient,
        )

        # 创建视觉形状（灰色半透明）
        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length / 2, thickness / 2, height / 2],
            rgbaColor=[0.5, 0.5, 0.5, 0.8], physicsClientId=self.physicsClient
        )

        # 创建墙体刚体
        wall_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=wall_collision,
            baseVisualShapeIndex=wall_visual,
            basePosition=[center_x, center_y, height / 2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, angle]),
            physicsClientId=self.physicsClient
        )

        # 设置物理参数（静态墙体）
        p.changeDynamics(
            wall_id,
            -1,
            lateralFriction=1.0,
            restitution=0.7,
            linearDamping=0.5,
            angularDamping=0.5, physicsClientId=self.physicsClient
        )
        self.walls.append(wall_id)

        return wall_id



    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)

        self._create_track()

        # 用当前 env 的独立 RNG 采样初始扰动；同一 seed 可严格复现。
        initial_position = np.asarray(self.start_pos, dtype=np.float64).copy()
        if self.start_position_noise:
            initial_position[:2] += self.np_random.uniform(
                -self.start_position_noise, self.start_position_noise, size=2
            )
        initial_heading = math.pi / 2
        if self.start_heading_noise:
            initial_heading += float(self.np_random.uniform(
                -self.start_heading_noise, self.start_heading_noise
            ))

        # 重置赛车位置和速度
        p.resetBasePositionAndOrientation(
            self.car,
            initial_position.tolist(),
            p.getQuaternionFromEuler([0, 0, initial_heading]),
            physicsClientId=self.physicsClient
        )
        p.resetBaseVelocity(
            self.car,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0], physicsClientId=self.physicsClient
        )

        self.current_step = 0
        self.last_pos = initial_position[:2].copy()

        #抽样录一个回合视频
        if self.recordVedio and imageio is not None and len(self.frames) > 10:
            os.makedirs("./logs", exist_ok=True)
            imageio.mimsave(f"./logs/racecar_{datetime.datetime.now().strftime('%H%M%S')}.mp4", self.frames, format='FFMPEG', fps=self.fps)
            self._log_scalar("steps/saveMP4", 1, self.total_step)

        self.frames = []
        if self.np_random.integers(0, 41) < 1:
            self.recordVedio = True
        else:
            self.recordVedio = False

        self._apply_action(0, 0.1)

        # 获取初始状态
        state = self._get_state()
        self.prev_state = state
        return state,{}

    def step(self, action):

        steer = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        steer = float(np.clip(steer, -1.0, 1.0))  # 转向 -1(左)到1(右)
        throttle = 0.1  # 油门 -1(倒车)到1(前进)

        # 设置赛车控制
        self._apply_action(steer, throttle)

        # 步进模拟
        p.stepSimulation(physicsClientId=self.physicsClient)
        self.current_step += 1
        self.total_step += 1

        # 获取新状态
        state = self._get_state()

        # 计算奖励
        external_reward, done = self._compute_reward(state)
        #记录回合结束的时候的位置
        if done:
            self._log_scalar("steps/pos_x", state[0], self.total_step)
            self._log_scalar("steps/pos_y", state[1], self.total_step)

        # 检查是否超过最大步数
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        if self.recordVedio:
            # 录制视频
            rgb = self._render_camera_frame()
            self.frames.append(rgb)


        reward = external_reward
        if self.total_step % 100 == 0:
            self._log_scalar("steps/external_reward", external_reward, self.total_step)


        info = {
            "steps": self.current_step,
            "position": state[:2],
            "is_success": done and external_reward > 0
        }
        if done:
            info["termination_reason"] = "collision" if external_reward < 0 else "goal"
        elif truncated:
            info["termination_reason"] = "time_limit"

        self.prev_state = state
        # 更新最后位置
        self.last_pos = state[:2]

        return state, reward, done, truncated, info

    def _apply_action(self, steer, throttle):
        """应用控制动作到赛车"""
        # 前轮转向
        steering_angle = steer   # 限制转向角度

        # 设置转向
        p.setJointMotorControl2(
            self.car,
            4,
            p.POSITION_CONTROL,
            targetPosition=-steering_angle, physicsClientId=self.physicsClient
        )
        p.setJointMotorControl2(
            self.car,
            6,
            p.POSITION_CONTROL,
            targetPosition=-steering_angle, physicsClientId=self.physicsClient
        )

        # 设置驱动轮速度
        max_force = 100
        target_vel = throttle * 50  # 控制速度


        for wheel in [2, 3, 5,7]:
            p.setJointMotorControl2(
                self.car,
                wheel,
                p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=max_force, physicsClientId=self.physicsClient
            )

    def _render_camera_frame(self):
        car_pos, _ = p.getBasePositionAndOrientation(self.car, physicsClientId=self.physicsClient)
        x, y = car_pos[0], car_pos[1]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[x - 5, y - 5, 5],
            cameraTargetPosition=[x, y, 0],
            cameraUpVector=[0, 0, 1]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=320 / 240,
            nearVal=0.1,
            farVal=100.0
        )

        width, height, rgb, _, _ = p.getCameraImage(
            width=320,
            height=240,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physicsClient
        )

        rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
        return rgb_array

    def _get_state(self):
        """获取当前状态（车辆位置、速度和转向关节状态）"""
        # 1. 获取车辆位置（x,y）
        pos, _ = p.getBasePositionAndOrientation(self.car, physicsClientId=self.physicsClient)
        position = np.array(pos[:2])  # 只取x,y坐标

        # 2. 获取车辆速度（x,y方向）
        linear_vel, _ = p.getBaseVelocity(self.car, physicsClientId=self.physicsClient)
        velocity = np.array(linear_vel[:2])  # 只取x,y方向速度

        # 3 获取小车的朝向：
        pos, orn = p.getBasePositionAndOrientation(self.car, physicsClientId=self.physicsClient)
        euler = p.getEulerFromQuaternion(orn)
        heading = euler[2]

        # 4. 获取转向关节状态（前轮两个关节）
        # 获取转向关节角度（前轮两个关节）
        steering_angle1 = p.getJointState(self.car, 4, physicsClientId=self.physicsClient)[0]
        steering_angle2 = p.getJointState(self.car, 6, physicsClientId=self.physicsClient)[0]

        # 获取转向关节角速度
        steering_vel1 = p.getJointState(self.car, 4, physicsClientId=self.physicsClient)[1]
        steering_vel2 = p.getJointState(self.car, 6, physicsClientId=self.physicsClient)[1]

        # 合并所有状态信息
        state = np.concatenate([
            position,  # 车辆位置 (x,y)
            velocity,  # 车辆速度 (vx,vy)
            [heading],
            [steering_angle1],  # 第一个转向关节角度
            [steering_angle2],  # 第二个转向关节角度
            [steering_vel1],  # 第一个转向关节角速度
            [steering_vel2]  # 第二个转向关节角速度
        ], dtype=np.float32)

        return state

    def _compute_reward(self, state):
        """计算奖励"""
        # 参考 bipedalwalker的设计，因为 bipedalwalker 这个task能够很好的收敛，所以参考它比较有信心

        done = False
        reward = 0

        # 1. 检查是否撞墙
        contact_points = p.getContactPoints(bodyA=self.car, physicsClientId=self.physicsClient)
        if contact_points:
            for point in contact_points:
                if point[2] in self.walls:  # 检查是否与墙壁碰撞, point[2]:body unique id of body B
                    reward =  -1 # 撞墙惩罚
                    done = True
                    self._log_scalar("steps/hitWall", 1, self.total_step)
                    return reward, done

        # 2. 检查是否到达终点
        finish_distance = np.linalg.norm(state[:2] - np.array(self.finish_pos[:2]))
        if finish_distance < 1.0:  # 接近终点
            reward = +3  # 到达终点奖励
            done = True
            self._log_scalar("steps/reachGoal", 1, self.total_step)
            return  reward, done
        # 3. 按照距离给出稠密的奖励
        else:
            reward = 1 / (1 + finish_distance)

        return reward, done

    def render(self, mode='human'):
        """渲染环境"""
        pass  # PyBullet会自动处理渲染

    def close(self):
        """关闭环境"""
        if self.physicsClient is not None and p.isConnected(self.physicsClient):
            p.disconnect(self.physicsClient)
            self.physicsClient = None

    def _log_scalar(self, tag, value, step):
        """写日志的进程安全边界：writer 应只属于当前进程。"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
