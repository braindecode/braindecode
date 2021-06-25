{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   {% for item in methods %}
   {%- if (item not in inherited_members and (
           not item.startswith('_') or
           item in ['__contains__', '__getitem__', '__iter__', '__len__', '__add__',
                    '__sub__', '__mul__', '__div__', '__neg__', '__hash__'])) %}
   .. automethod:: {{ item }}

   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. include:: {{fullname}}.examples

.. raw:: html

    <div style='clear:both'></div>
