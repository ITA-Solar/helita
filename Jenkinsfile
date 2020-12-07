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
from helita.sim.bifrost import BifrostData, Bifrost_units 
from helita.sim.ebysus import EbysusData 
import helita.sim.muram
import helita.sim.cipmocct 
import helita.sim.laresav
import helita.sim.radyn
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
