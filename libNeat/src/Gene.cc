#include "Gene.hh"
#include "Link.hh"
#include "Node.hh"
#include <memory>
#include <iostream>

Gene::Gene(std::shared_ptr<Node>& origin_node,
           std::shared_ptr<Node>& dest_node,
	   double weight,
	   Gene::Status s,
           int ninnovation)
  : connection(std::make_shared<Link>(origin_node,dest_node))
  , innovation(ninnovation)
  , status(s) {

  // fill the participant nodes link lists
  origin_node->output_links.push_back(connection);
  dest_node->input_links.push_back(connection);
}
