### MMCV
- [ ] Implement the attr 'get' of 'Config'
- [ ] Config bugs: None type to '{}' with addict
- [ ] Default logger should be only with gpu0
- [ ] Unit Test: mmcv and mmcv.torchpack


### MMDetection

#### Basic
- [ ] Implement training function without distributed
- [ ] Verify nccl/nccl2/gloo
- [ ] Replace UGLY code: params plug in 'args' to reach a global flow
- [ ] Replace 'print' by 'logger'


#### Testing
- [ ] Implement distributed testing
- [ ] Implement single gpu testing


#### Refactor
- [ ] Re-consider params names
- [ ] Refactor functions in 'core'
- [ ] Merge single test & aug test as one function, so as other redundancy

#### New features
- [ ] Plug loss params into Config
- [ ] Multi-head communication
