# github actions

eval "$(ssh-agent -s)"
ssh-add /scratch/pawsey0411/fchen1/.ssh/setonix
---------------------------------------
source /scratch/pawsey0411/fchen1/DeepPySR/.venv//bin/activate
export JULIA_DEPOT_PATH="/scratch/pawsey0411/fchen1/.julia_depot"
export PYTHON_JULIAPKG_OFFLINE=no
rm -rf /scratch/pawsey0411/fchen1/DeepPySR/.venv/julia_env
python -m juliapkg update
python -m juliacall.check



# notes:
stroke, diabetes, diabetes130us are 100 nit, others are 500

| Dataset          | Machine  | nit | Status    |
|:-----------------|:---------|:----|:----------|
| BMI              | setonix  | 500 | running   |
| BMI warm         | setonix  | 500 | running   |
| feynman          | setonix  | 500 | running   |
| feynman warm     | setonix  | 500 | running   |
| heart            | setonix  | 500 | completed |
| heart warm       | setonix  | 500 | running   |
| stroke           | setonix  | 100 | running   |
| stroke warm      | setonix  | 100 | running   |
| bodyfat          | setonix  | 500 | completed |
| bodyfat warm     | setonix  | 500 | running   |
| wine             | setonix  | 500 | running   |
| wine warm        | setonix  | 500 | running   |
| diabetes         | a400     | 100 | completed |
| diabetes warm    | setonix  | 100 | running   |
| diabetes130      | dia-ml s | 100 | running   |
| diabetes130 warm | setonix  | 100 | running   |
