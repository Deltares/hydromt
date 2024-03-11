# Example Model Usage

```py
m = Model()
m.add_component("grid", GridComponent)

m.build(config)

# or

m.update(config)

##########################

m = Model()
g:GridComponent = m.add_component("grid", GridComponent)

g.set(...)
g.create(...)

g2:GridComponent = m.get_component("grid", type(GridComponent))
g2.set(...)

m.grid.set(...)

# or

m.grid.create(...)

m.grid.write()
m.grid.read()
```

```yaml
components:
    grid:
        type: GridComponent
        filename: grid.nc
    subgrid:
        type: GridComponent
        filename: subgrid.nc
steps:
    - step: create
      on: grid
      with:
        shape: [10, 10]
        dtype: float32
        fill: 0.0
    - step: create
      on: subgrid
      with:
        shape: [10, 10]
        dtype: float32
        fill: 0.0
    # - step: create
    #   on: grid
    #   data: /path/to/data
```
