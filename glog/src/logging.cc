#include "logging.h"

namespace google {

// This is the set of log sinks. This must be in a separate library to ensure
// that there is only one instance of this across the entire program.
std::set<google::LogSink *> log_sinks_global;

int log_severity_global(INFO);

}
