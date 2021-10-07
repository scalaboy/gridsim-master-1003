class Agent(object):
    def __init__(self, settings, this_directory_path):
        self.num_gen = settings.num_gen

    def act(self, obs, reward, done=False):
        adjust_gen_p_action_space = obs.action_space['adjust_gen_p']
        adjust_gen_v_action_space = obs.action_space['adjust_gen_v']

        adjust_gen_p = adjust_gen_p_action_space.sample()
        adjust_gen_v = adjust_gen_v_action_space.sample()

        action = {'adjust_gen_p': adjust_gen_p, 'adjust_gen_v': adjust_gen_v}
        return action