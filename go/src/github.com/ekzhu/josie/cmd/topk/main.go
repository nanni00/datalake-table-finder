package main

import (
	"database/sql"
	"flag"
	"fmt"
	//"path/filepath"

	"github.com/ekzhu/josie"
)

var (
	pgServer, pgPort, pgDatabase 	string
	// benchmark        				string
	output           				string
	cpuProfile       				bool
	k								int
)

func main() {
	flag.StringVar(&pgServer, "pg-server", "localhost", "Postgres server addresss")
	flag.StringVar(&pgDatabase, "pg-database", "sloth500", "Postgres database name")
	flag.StringVar(&pgPort, "pg-port", "5442", "Postgres server port")
	// flag.StringVar(&benchmark, "benchmark", 	"sloth500", "The name of the benchmark dataset to use")
	flag.IntVar(&k, "k", 3, "The k value for the topK search")
	flag.StringVar(&output, "output", 			"results", "Output directory for results")
	flag.BoolVar(&cpuProfile, "cpu-profile", false, "Enable CPU profiling")
	flag.Parse()
	db, err := sql.Open("postgres", fmt.Sprintf("host=%s port=%s dbname=%s sslmode=disable", pgServer, pgPort, pgDatabase))
	if err != nil {
		panic(err)
	}
	defer db.Close()
	
	joise.NanniExperiments(db, k, output, cpuProfile, true)
	
	// if benchmark == "canada_us_uk" {
	// 	joise.RunOpenDataExperiments(db, filepath.Join(output, benchmark), cpuProfile, true)
	// }
	// if benchmark == "webtable" {
	// 	joise.RunWebTableExperiments(db, filepath.Join(output, benchmark), cpuProfile, true)
	// }
}
