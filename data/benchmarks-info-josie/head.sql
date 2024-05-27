--
-- PostgreSQL database dump
--

-- Dumped from database version 10.0
-- Dumped by pg_dump version 10.0

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: webtable_inverted_lists; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE webtable_inverted_lists (
    token integer NOT NULL,
    frequency integer NOT NULL,
    duplicate_group_id integer NOT NULL,
    duplicate_group_count integer NOT NULL,
    set_ids integer[] NOT NULL,
    set_sizes integer[] NOT NULL,
    match_positions integer[] NOT NULL,
    raw_token bytea NOT NULL
);


--
-- Name: webtable_queries_100; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE webtable_queries_100 (
    id integer,
    tokens integer[]
);


--
-- Name: webtable_queries_10k; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE webtable_queries_10k (
    id integer,
    tokens integer[]
);


--
-- Name: webtable_queries_1k; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE webtable_queries_1k (
    id integer,
    tokens integer[]
);


--
-- Name: webtable_sets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE webtable_sets (
    id integer NOT NULL,
    size integer NOT NULL,
    num_non_singular_token integer NOT NULL,
    tokens integer[] NOT NULL
);


--
-- Data for Name: webtable_inverted_lists; Type: TABLE DATA; Schema: public; Owner: -
--

COPY webtable_inverted_lists (
token,      frequency,  duplicate_group_id, duplicate_group_count,  set_ids,    set_sizes,  match_positions,    raw_token) FROM stdin;
80474718	1	        30430758	        15	                    {95962860}	{29}	    {14}	            \\x657269635f6861726f6c645f636f6c6c6965725f646965645f6d61725f31355f313936355f73745f6c756b65735f686f6d652c5f73745f6a6f686e732c5f6275726965645f666f726573745f72645f616e675f63656d2e5f6d645f656c7369655f77616c6b65725f28625f313839325f2d5f645f6a756c795f31365f313936345f73745f6a6f686e73292e5f272777616c6b657227275f69735f66726f6d5f6865725f6865616473746f6e653b5f646174655f6f665f64656174685f286275726965645f61735f656c7369655f636f6c6c696572295f69735f66726f6d5f616e675f63656d5f62757269616c5f6c65646765722e
179945123	24	        24197146	        1	                    {20782557,37556401,42447827,47447109,60557910,77334833,78414533,101807374,107158224,117392783,120310178,129399287,130946645,132938481,136012946,136951767,156330350,190755669,200362168,214007548,226102198,232542674,233228883,245436071}	{170,65,170,65,65,169,66,67,66,170,170,65,65,171,65,65,66,169,65,170,66,65,65,169}	{71,33,71,33,33,71,31,32,31,71,71,33,33,72,33,33,31,71,33,71,31,33,33,71}	\\x742d6d6f62696c6527735f66697273745f687370612b5f6d6f64656d5f676f65735f6f6e5f73616c655f73756e646179


2820883	    1	1069632	    10	{162331178}	{10}	{6}	\\x68747470733a2f2f6865632e73752f62657365
64733288	1	24474069	3	{84593750}	{10}	{0}	\\x6e65775f626173696e5f776179
79200238	1	29946057	10	{214905658}	{10}	{3}	\\x31305f72616e646572736f6e5f64726976652c5f6d6578626f726f7567682c5f736f7574685f796f726b73686972652c5f7336345f357577
147884208	3	899170	    137	{153079555,174993712,244676853}	{155,155,155}	{101,101,101}	\\x312c3337382c3335342c353833
156412633	3	38509220	1	{89532176,142643557,182076676}	{10,561,113}	{5,258,30}	\\x6661776e62726f6f6b
54301153	1	20526502	12	{20856936}	{14}	{5}	\\x5b6e635f64656174685f636f6c6c656374696f6e5f7374617465735f64656174685f61735f6a756e655f312c5f323030325d
165673083	5	26347087	1	{21618495,42722048,115020669,221720625,251106473}	{48,54,45,49,49}	{41,45,36,40,40}	\\x72653a5f646f63732c5f7365636f6e645f747279
149940998	3	9946324	    1	{13113545,32053446,117292668}	{10,14,14}	{7,7,7}	\\x6c27616d6f75725f656d706f7274655f746f7574
177295593	15	2707077	    1	{28269056,45524591,47753668,70974364,73551551,112978226,117575062,196146941,205298590,207478381,208678290,228309367,234312288,248788621,249207175}	{264,154,328,267,197,200,208,71,222,433,554,228,194,225,205}	{163,78,213,145,105,118,119,28,148,296,380,132,101,152,149}	\\x6869676865725f65642d6679323030385f66756e64696e67
86958368	1	32921305	9	{26661960}	{9}	{6}	\\x6973757a755f696d70756c73655f636f776c5f686f6f64



