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
	test_tag        				string
	outputDir           			string
	resultsFile						string
	cpuProfile       				bool
	useMemTokenTable				bool
	k								int
	verbose							bool
)

func main() {
	flag.StringVar(&pgServer, 	"pg-server", 	"localhost", 	"Postgres server addresss")
	flag.StringVar(&pgDatabase, "pg-database", 	"nanni",	 	"Postgres database name")
	flag.StringVar(&pgPort, 	"pg-port", 		"5442", 		"Postgres server port")
	flag.StringVar(&test_tag, 	"test_tag", 	"n45673_mset", 	"The name of the benchmark dataset to use")
	flag.IntVar(&k, 			"k", 			3, 				"The k value for the topK search")
	flag.StringVar(&outputDir, 	"outputDir", 	"results", 		"Output directory for results")
	flag.StringVar(&resultsFile,"resultsFile",	"resultsFile",	"The file where the final results will be stored")
	flag.BoolVar(&useMemTokenTable, "useMemTokenTable", true, "")
	flag.BoolVar(&cpuProfile, 	"cpu-profile", 	false, 			"Enable CPU profiling")
	flag.BoolVar(&verbose, "verbose", false, "")
	flag.Parse()
	
	db, err := sql.Open("postgres", fmt.Sprintf("host=%s port=%s dbname=%s sslmode=disable", pgServer, pgPort, pgDatabase))
	if err != nil {
		panic(err)
	}
	defer db.Close()
	
	joise.NanniExperiments(db, k, test_tag, outputDir, resultsFile, cpuProfile, useMemTokenTable, verbose)
	
	// if benchmark == "canada_us_uk" {
	// 	joise.RunOpenDataExperiments(db, filepath.Join(output, benchmark), cpuProfile, true)
	// }
	// if benchmark == "webtable" {
	// 	joise.RunWebTableExperiments(db, filepath.Join(output, benchmark), cpuProfile, true)
	// }
}
