./gradlew distDocker
cd ansible
# For fresh DB Deploy
ansible-playbook -i environments/local couchdb.yml
ansible-playbook -i environments/local initdb.yml
# Only on fresh builds
ansible-playbook -i environments/local wipe.yml
ansible-playbook -i environments/local openwhisk.yml

# installs a catalog of public packages and actions
ansible-playbook -i environments/local postdeploy.yml

# to use the API gateway
ansible-playbook -i environments/local apigateway.yml
ansible-playbook -i environments/local routemgmt.yml
