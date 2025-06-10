# Metaflow Cheat Sheet

## Basic Commands

### Running Flows
```bash
# Run a flow
python my_flow.py run

# Run with parameters
python my_flow.py run --test_size 0.3 --random_state 123

# Show flow information
python my_flow.py show

# List all runs
metaflow list runs MyFlow

# Show specific run
metaflow show MyFlow/123
```

### Flow Structure
```python
from metaflow import FlowSpec, step, Parameter

class MyFlow(FlowSpec):
    
    # Parameters
    param = Parameter('param', default='value')
    
    @step
    def start(self):
        # Initialize
        self.data = "some data"
        self.next(self.process)
    
    @step
    def process(self):
        # Process data
        self.result = self.data.upper()
        self.next(self.end)
    
    @step
    def end(self):
        # Finalize
        print(f"Result: {self.result}")

if __name__ == '__main__':
    MyFlow()
```

### Accessing Results
```python
from metaflow import Flow

# Get flow
flow = Flow('MyFlow')

# Get latest run
run = flow.latest_run

# Access step data
data = run['step_name'].task.data.artifact_name

# Check if run was successful
if run.successful:
    print("Run completed successfully")
```

### Parallel Execution
```python
@step
def start(self):
    self.items = ['a', 'b', 'c']
    self.next(self.process, foreach='items')

@step
def process(self):
    # This runs in parallel for each item
    self.result = self.input.upper()
    self.next(self.join)

@step
def join(self, inputs):
    # Merge parallel results
    self.all_results = [inp.result for inp in inputs]
    self.next(self.end)
```

### Useful Decorators
```python
from metaflow import step, catch, retry, timeout

@catch(var='error_info')
@retry(times=3)
@timeout(seconds=3600)
@step
def robust_step(self):
    # Step with error handling
    pass
```

## Best Practices

1. **Use descriptive step names**
2. **Store intermediate results as artifacts**
3. **Add docstrings to flows and steps**
4. **Use parameters for configurable values**
5. **Handle errors gracefully with @catch**
6. **Version your flows in git**