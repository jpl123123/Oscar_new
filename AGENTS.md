# NPU ownership

The user authorizes only physical Ascend devices **4,5,6,7**. Devices 0,1,2,3
belong to other users and must not be used by this project.

Every launch, calibration, and NPU test script must unconditionally set
`export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` before running Python or importing
NPU libraries. Python-only launchers and calibration child processes must
enforce the same value before importing TorchNPU/vLLM. Do not make this an
overridable default. Tests must cover overriding an inherited conflicting value.

Under this visibility mapping, logical NPU indices 0..3 refer to authorized
physical devices 4..7. Do not confuse logical tensor-parallel ranks with physical
card IDs. Never stop or modify workloads running on physical devices 0..3.

Keep `references/` read-only. All adaptation code belongs to the external package.
