/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  is_initialized = true;
  num_particles = 5;

  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_t(theta, std[2]);

  for (int i=0; i<num_particles; i++) { // initialize individual particles
  	Particle p;
  	p.id = i;
	p.x = dist_x(gen);
	p.y = dist_y(gen);
	p.theta = dist_t(gen);
	p.weight = 1.0;
	particles.push_back(p);
	weights.push_back(1);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  for (int i=0; i<num_particles; i++) {
  	double x = particles[i].x;
  	double y = particles[i].y;
  	double theta = particles[i].theta;

  	if (fabs(yaw_rate) < 0.0001) {
  		x += velocity * delta_t * cos(theta);
  		y += velocity * delta_t * sin(theta);
  	}
  	else {
  		x += (velocity/yaw_rate)*(sin(theta + yaw_rate * delta_t) - sin(theta));
  		y += (velocity/yaw_rate)*(cos(theta) - cos(theta + yaw_rate * delta_t));
  		theta += yaw_rate * delta_t;
  	}

    normal_distribution<double> dist_x(x, std_pos[0]);
  	normal_distribution<double> dist_y(y, std_pos[1]);
  	normal_distribution<double> dist_t(theta, std_pos[2]);

  	particles[i].x = dist_x(gen);
  	particles[i].y = dist_y(gen);
  	particles[i].theta = dist_t(gen);
  	while (particles[i].theta >= 2*M_PI) {particles[i].theta -= 2*M_PI;}
  	while (particles[i].theta <= 0) {particles[i].theta += 2*M_PI;}
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {


  // test each particle using the observations
  for (int i=0; i<num_particles; i++) {
  	Particle part;
  	part.id = i;
  	part.x = particles[i].x;
  	part.y = particles[i].y;
  	part.theta = particles[i].theta;
  	part.weight = 1.0;
  	vector<int> assoc;
  	vector<double> sense_x;
  	vector<double> sense_y;

  	//translate vehicle coordinate observations into map coordinates
  	vector<LandmarkObs> map_predictions;
  	for (int j=0; j<observations.size(); j++) {
	  LandmarkObs obs, map_trans;
	  obs = observations[j];

	  map_trans.id = 0;
	  map_trans.x = part.x + (obs.x * cos(part.theta) - obs.y * sin(part.theta));
	  map_trans.y = part.y + (obs.x * sin(part.theta) + obs.y * cos(part.theta));
	  map_predictions.push_back(map_trans);
	}

    // for each observation calculate the gaussian weight and apply it to the particle weight
    for (int j=0; j<map_predictions.size(); j++) {
      double distance = 1000000.0;
      double dist_check = 0.0;
      for (int k=0; k<map_landmarks.landmark_list.size(); k++) {
      	dist_check = dist(map_predictions[j].x, map_predictions[j].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
      	if (dist_check < distance) {
      		distance = dist_check;
      		map_predictions[j].id = k;
      	}
      }
  	  double exp_x = ((pow((map_predictions[j].x - map_landmarks.landmark_list[map_predictions[j].id].x_f),2)/pow(std_landmark[0],2)));
  	  double exp_y = ((pow((map_predictions[j].y - map_landmarks.landmark_list[map_predictions[j].id].y_f),2)/pow(std_landmark[1],2)));
      double norm = 1/(2*M_PI*std_landmark[0]*std_landmark[0]);
  	  part.weight *= norm * exp(-(exp_x + exp_y));
  	  assoc.push_back(map_landmarks.landmark_list[map_predictions[j].id].id_i);
  	  sense_x.push_back(map_predictions[j].x);
      sense_y.push_back(map_predictions[j].y);
  	}
    particles[i] = SetAssociations(part, assoc, sense_x, sense_y);
    weights[i] = part.weight;
  }
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	default_random_engine gen;
	discrete_distribution<int> dist_p(weights.begin(), weights.end());

	vector<Particle> new_particles;
	for (int i=0; i<num_particles; i++) {
		new_particles.push_back(particles[dist_p(gen)]);
		new_particles[i].id = i;
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
