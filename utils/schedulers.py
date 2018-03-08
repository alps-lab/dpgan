

def get_scheduler(name, config):
    if name == "basic":
        from dp.schedulers.basic import BasicScheduler
        return BasicScheduler(config.critic_iters)

    raise Exception("unsupported scheduler: %s." % name)