"""
Functions common to Posterior and Likelihood classes.
TODO: I need to derive them somehow from a ComposedISDCallable (or similar) class 
"""

def _get_component_var_param_types(self):

    return {var: c._var_param_types[var] for c in self._components.values() 
            for var in c.variables if not var == 'mock_data'}

def _setup_variables(self):

    for c in self._components.values():
        for var in c.variables:
            if not var in self.variables and not var == 'mock_data':
                self._register_variable(var, var in c.differentiable_variables)

def _setup_parameters(self):

    for c in self._components.values():
        for p in c.get_params():
            if not p in self.parameters:
                self._register(p.name)
                self[p.name] = p.__class__(p.value, p.name)
            p.bind_to(self[p.name])
                
def fix_variables(self, **fixed_vars):

    for var, value in fixed_vars.items():
        self._register(var)
        self._delete_variable(var)
        self[var] = self._var_param_types[var](value, var)
        for c in self._components.values():
            if var in c.variables:
                c.fix_variables(**{var: value})
                c[var].bind_to(self[var])
