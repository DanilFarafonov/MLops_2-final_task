from clearml import Task
from clearml.automation import (
    UniformParameterRange,
    HyperParameterOptimizer,
)

task = Task.init(project_name='MLOps_2.4',
                 task_name='optimize_svc_HP',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

args = {
    'template_task_id': "d708e5d035bc41f9b1f0bf71347bf40a",
    'run_as_service': False,
}

an_optimizer = HyperParameterOptimizer(
    base_task_id=args['template_task_id'],
    hyper_parameters=[
        UniformParameterRange('General/C', min_value=0.5, max_value=2.6, step_size=0.5),
        UniformParameterRange('General/gamma', min_value=0.5, max_value=1.2, step_size=0.2)
    ],
    objective_metric_title='F1_score',
    objective_metric_series='F1',
    objective_metric_sign='max'
)

an_optimizer.start_locally()
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
task.upload_artifact('top_exp', top_exp)
an_optimizer.wait()
an_optimizer.stop()
