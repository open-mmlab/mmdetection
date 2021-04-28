#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

from noussdk.communication.grpc.grpc_server import GRPCServer
from noussdk.entities.task_environment import TaskEnvironment
from noussdk.usecases.tasks.interfaces.task_interface import ITask
from tasks.mmdetection_tasks.detection import MMObjectDetectionTask


class MMDetectionServer(GRPCServer):
    task: MMObjectDetectionTask

    def get_task_type(self):
        return MMObjectDetectionTask

    def initialize_task(self, task_environment: TaskEnvironment) -> ITask:
        self.task = MMObjectDetectionTask(task_environment)
        return self.task


port = 50067
mmdetection_server = MMDetectionServer()
mmdetection_server.serve(port)
