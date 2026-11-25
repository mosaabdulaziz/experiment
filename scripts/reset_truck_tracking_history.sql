-- Clears the derived tracking history (sightings, visits, journeys and what hangs off
-- them) of ONE company in a local truck-tracking database, and resets its trucks'
-- last-seen state. Trucks, stops and cameras are kept. The feed bridge replays every
-- vehicle the tracker knows the next time the app restarts, rebuilding the history.
-- Add `-v drop_trucks=1` to also delete the trucks (plates) themselves.
-- usage: psql -v company="'<company uuid>'" [-v drop_trucks=1] -f scripts/reset_truck_tracking_history.sql
begin;
with t as (select id from trucks where company_id = :company::uuid),
     j as (select id from journeys where company_id = :company::uuid)
select (select count(*) from t) trucks, (select count(*) from j) journeys,
       (select count(*) from visits where company_id = :company::uuid) visits,
       (select count(*) from sightings where company_id = :company::uuid) sightings;
delete from visit_transitions where truck_id in (select id from trucks where company_id = :company::uuid);
delete from journey_paths where truck_id in (select id from trucks where company_id = :company::uuid);
delete from visit_loops where truck_id in (select id from trucks where company_id = :company::uuid);
delete from journey_step_results where journey_id in (select id from journeys where company_id = :company::uuid);
delete from journey_step_progress where journey_id in (select id from journeys where company_id = :company::uuid);
delete from journey_step_gaps where journey_id in (select id from journeys where company_id = :company::uuid);
delete from journey_extra_stops where journey_id in (select id from journeys where company_id = :company::uuid);
delete from visits where company_id = :company::uuid;
delete from journeys where company_id = :company::uuid;
delete from sightings where company_id = :company::uuid;
update trucks set last_seen_at = null, last_stop_id = null, last_branch_id = null where company_id = :company::uuid;
\if :{?drop_trucks}
delete from truck_aliases where truck_id in (select id from trucks where company_id = :company::uuid);
delete from truck_assignment_history where truck_id in (select id from trucks where company_id = :company::uuid);
delete from truck_schedules where truck_id in (select id from trucks where company_id = :company::uuid);
delete from trucks where company_id = :company::uuid;
\endif
commit;
