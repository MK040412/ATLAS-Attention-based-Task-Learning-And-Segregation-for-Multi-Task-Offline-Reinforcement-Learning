import numpy as np
import torch

class ImprovedRewardSystem:
    """개선된 보상 시스템 - Dense Reward + Curriculum"""
    
    def __init__(self):
        # 보상 가중치
        self.w_distance = 1.0      # 목표까지 거리
        self.w_approach = 0.5      # 큐브에 접근
        self.w_contact = 2.0       # 큐브와 접촉
        self.w_push = 3.0          # 큐브 밀기
        self.w_goal = 10.0         # 목표 달성
        self.w_efficiency = 0.1    # 효율성 (에너지 절약)
        
        # 이전 상태 저장
        self.prev_cube_to_goal_dist = None
        self.prev_ant_to_cube_dist = None
        self.contact_steps = 0
        self.total_steps = 0
        
    def reset(self):
        """에피소드 시작 시 리셋"""
        self.prev_cube_to_goal_dist = None
        self.prev_ant_to_cube_dist = None
        self.contact_steps = 0
        self.total_steps = 0
    
    def compute_reward(self, ant_pos, cube_pos, goal_pos, action, info=None):
        """개선된 보상 계산"""
        self.total_steps += 1
        
        # 기본 거리들
        cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)
        ant_to_cube_dist = np.linalg.norm(ant_pos - cube_pos)
        ant_to_goal_dist = np.linalg.norm(ant_pos - goal_pos)
        
        reward = 0.0
        reward_info = {}
        
        # 1. 목표 달성 보상 (가장 중요)
        if cube_to_goal_dist < 0.3:  # 목표 근처
            goal_reward = self.w_goal * (1.0 - cube_to_goal_dist / 0.3)
            reward += goal_reward
            reward_info['goal_reward'] = goal_reward
        
        # 2. 큐브를 목표로 밀기 보상 (진행 상황 기반)
        if self.prev_cube_to_goal_dist is not None:
            progress = self.prev_cube_to_goal_dist - cube_to_goal_dist
            push_reward = self.w_push * progress * 10  # 진행도에 따른 보상
            reward += push_reward
            reward_info['push_reward'] = push_reward
        
        # 3. 큐브에 접근 보상
        if self.prev_ant_to_cube_dist is not None:
            approach_progress = self.prev_ant_to_cube_dist - ant_to_cube_dist
            approach_reward = self.w_approach * approach_progress * 5
            reward += approach_reward
            reward_info['approach_reward'] = approach_reward
        
        # 4. 접촉 보상 (큐브 근처에 있을 때)
        if ant_to_cube_dist < 0.5:  # 접촉 범위
            contact_reward = self.w_contact * (1.0 - ant_to_cube_dist / 0.5)
            reward += contact_reward
            reward_info['contact_reward'] = contact_reward
            self.contact_steps += 1
        
        # 5. 거리 기반 기본 보상 (항상 제공)
        distance_reward = -self.w_distance * cube_to_goal_dist / 10.0
        reward += distance_reward
        reward_info['distance_reward'] = distance_reward
        
        # 6. 효율성 보상 (에너지 절약)
        action_penalty = -self.w_efficiency * np.sum(np.abs(action))
        reward += action_penalty
        reward_info['efficiency_reward'] = action_penalty
        
        # 7. 탐험 보너스 (초기 단계)
        if self.total_steps < 100:
            exploration_bonus = 0.1 * (100 - self.total_steps) / 100
            reward += exploration_bonus
            reward_info['exploration_bonus'] = exploration_bonus
        
        # 8. 시간 페널티 (너무 오래 걸리면)
        if self.total_steps > 500:
            time_penalty = -0.01 * (self.total_steps - 500)
            reward += time_penalty
            reward_info['time_penalty'] = time_penalty
        
        # 이전 상태 업데이트
        self.prev_cube_to_goal_dist = cube_to_goal_dist
        self.prev_ant_to_cube_dist = ant_to_cube_dist
        
        # 총 보상 정보
        reward_info['total_reward'] = reward
        reward_info['cube_to_goal_dist'] = cube_to_goal_dist
        reward_info['ant_to_cube_dist'] = ant_to_cube_dist
        reward_info['contact_ratio'] = self.contact_steps / max(1, self.total_steps)
        
        return reward, reward_info

class CurriculumManager:
    """커리큘럼 학습 관리자"""
    
    def __init__(self):
        self.current_level = 0
        self.success_threshold = 0.7  # 70% 성공률
        self.min_episodes = 20        # 최소 에피소드 수
        self.episode_results = []
        
        # 커리큘럼 레벨 정의
        self.curriculum_levels = [
            {
                'name': 'Level 1: Close Push',
                'cube_position': [1.5, 0.0, 0.1],
                'goal_position': [2.0, 0.0, 0.1],
                'success_distance': 0.3,
                'max_steps': 300,
                'description': 'Push cube a short distance'
            },
            {
                'name': 'Level 2: Medium Push', 
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 0.0, 0.1],
                'success_distance': 0.3,
                'max_steps': 500,
                'description': 'Push cube medium distance'
            },
            {
                'name': 'Level 3: Long Push',
                'cube_position': [1.0, 0.0, 0.1],
                'goal_position': [4.0, 0.0, 0.1], 
                'success_distance': 0.3,
                'max_steps': 800,
                'description': 'Push cube long distance'
            },
            {
                'name': 'Level 4: Diagonal Push',
                'cube_position': [1.5, 1.0, 0.1],
                'goal_position': [2.5, 2.0, 0.1],
                'success_distance': 0.3,
                'max_steps': 600,
                'description': 'Push cube diagonally'
            },
            {
                'name': 'Level 5: Complex Navigation',
                'cube_position': [1.0, 1.5, 0.1],
                'goal_position': [4.0, -1.5, 0.1],
                'success_distance': 0.3,
                'max_steps': 1000,
                'description': 'Complex push with navigation'
            }
        ]
    
    def get_current_level(self):
        """현재 레벨 정보 반환"""
        if self.current_level < len(self.curriculum_levels):
            return self.curriculum_levels[self.current_level]
        else:
            # 마지막 레벨 반복
            return self.curriculum_levels[-1]
    
    def add_episode_result(self, success, final_distance, steps, total_reward):
        """에피소드 결과 추가"""
        result = {
            'success': success,
            'final_distance': final_distance,
            'steps': steps,
            'total_reward': total_reward,
            'level': self.current_level
        }
        self.episode_results.append(result)
        
        # 최근 결과만 유지 (메모리 절약)
        if len(self.episode_results) > 100:
            self.episode_results = self.episode_results[-100:]
    
    def should_advance_level(self):
        """다음 레벨로 진행할지 결정"""
        if len(self.episode_results) < self.min_episodes:
            return False
        
        # 현재 레벨의 최근 결과들만 확인
        current_level_results = [r for r in self.episode_results[-self.min_episodes:] 
                               if r['level'] == self.current_level]
        
        if len(current_level_results) < self.min_episodes:
            return False
        
        # 성공률 계산
        success_rate = sum(r['success'] for r in current_level_results) / len(current_level_results)
        
        return success_rate >= self.success_threshold
    
    def advance_level(self):
        """다음 레벨로 진행"""
        if self.current_level < len(self.curriculum_levels) - 1:
            self.current_level += 1
            print(f"🎓 Advanced to {self.curriculum_levels[self.current_level]['name']}")
            return True
        else:
            print("🏆 Curriculum completed! Continuing with final level.")
            return False
    
    def get_statistics(self):
        """학습 통계 반환"""
        if not self.episode_results:
            return {}
        
        recent_results = self.episode_results[-20:]  # 최근 20개 에피소드
        current_level_results = [r for r in recent_results if r['level'] == self.current_level]
        
        stats = {
            'current_level': self.current_level,
            'level_name': self.get_current_level()['name'],
            'total_episodes': len(self.episode_results),
            'recent_success_rate': sum(r['success'] for r in recent_results) / max(1, len(recent_results)),
            'current_level_episodes': len(current_level_results),
            'current_level_success_rate': sum(r['success'] for r in current_level_results) / max(1, len(current_level_results)),
            'average_steps': np.mean([r['steps'] for r in recent_results]) if recent_results else 0,
            'average_reward': np.mean([r['total_reward'] for r in recent_results]) if recent_results else 0
        }
        
        return stats