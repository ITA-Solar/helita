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
      stage('test import') {
         steps {
            sh '''#!/bin/csh 
            python 
            from helita.sim.bifrost import BifrostData as br
            from helita.sim.ebysus import EbysusData as eb
            from helita.sim.bifrost import BifrostUnits as uni
            exit	  
'''
         }
      }
   }
}
