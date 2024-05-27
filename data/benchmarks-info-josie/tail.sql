132351008	1	1	{184615401}
117047903	15	15	{184190044,184215603,184261293,184294717,184301734,184364125,184374133,184454887,184518878,184570816,184579648,184602027,184608125,184611744,184639139}
27414223	1	1	{183744543}
127423078	4	4	{184644336,184644373,184644408,184644468}
10878913	3	3	{181793318,181793319,181793320}
\.


--
-- Name: webtable_inverted_lists_token_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX webtable_inverted_lists_token_idx ON webtable_inverted_lists USING btree (token);


--
-- Name: webtable_sets_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX webtable_sets_id_idx ON webtable_sets USING btree (id);


--
-- PostgreSQL database dump complete
--

