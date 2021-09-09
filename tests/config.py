try:
    import os
    from e2e import config as config_e2e

    config_e2e.repository_name = os.environ.get("TT_REPOSITORY_NAME", "ote/training_extensions/external/mmdetection")
except ImportError:
    pass
