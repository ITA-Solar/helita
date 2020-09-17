pipeline {
   agent any
   stages {
      stage('update') {
         steps {
            sh '''#!/bin/csh 
            echo hello jenkins
            echo $PATH
            python setup.py develop --user 
'''
         }
      }
         stage('create test py') {
         steps {
            sh '''#!/bin/csh 
            cat <<-'TEST_CASES' > test.py
#!/usr/bin/env python
from helita.sim.bifrost import BifrostData as br
from helita.sim.ebysus import EbysusData as eb
from helita.sim.bifrost import Bifrost_units as uni            
'''
         }
      }
      stage('running test py') {
         steps {
            sh '''#!/bin/csh 
            python -m test.py || [[ $? -eq 1 ]]
            exit	  
'''
         }
      }
   }
}