# Autoriser les gros notebooks
c.NotebookApp.max_buffer_size = 2_000_000_000  # 2GB

# Autoriser les gros fichiers uploadés
c.NotebookApp.tornado_settings = {
    "max_buffer_size": 2_000_000_000  # 2GB
}

# Timeout long pour les kernels lourds
c.MappingKernelManager.kernel_info_timeout = 60
c.MappingKernelManager.buffer_offline_timeout = 60

# Autoriser les gros messages entre kernel et frontend
c.NotebookApp.iopub_msg_rate_limit = 200000
c.NotebookApp.iopub_data_rate_limit = 1e10
c.ServerApp.rate_limit_window=5.0 #(sec)

# Autoriser les gros WebSockets
# c.NotebookApp.websocket_max_message_size = 2_000_000_000  # 2GB

# Désactiver le check d'origine derrière un reverse proxy
c.NotebookApp.allow_origin = '*'
c.NotebookApp.allow_remote_access = True

# Adresse et port
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False

