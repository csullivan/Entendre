#include "Gene.hh"
#include "Link.hh"
#include "Node.hh"
#include <memory>

Gene::Gene(std::shared_ptr<Node>& inode,
           std::shared_ptr<Node>& onode,
	   double weight,
	   Gene::Status s,
           int ninnovation)
  : connection(std::make_shared<Link>(inode,onode))
  , innovation(ninnovation)
  , status(s) {

}
