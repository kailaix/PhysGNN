include("common.jl")
reset_default_graph()
sess = Session(); init(sess)

env = Environment(sess)

db = Database("result.db")
execute(db, """
CREATE TABLE exact (
    obs real,
    mu real
)
""")
for i = 1:10000
    μ = sample_exact(env)
    obs = env(μ)
    params = [(μ[i], obs[i]) for i = 1:length(μ)]
    execute(db, """
INSERT INTO exact VALUES (?,?)
""", params)
end

res = execute(db, """
SELECT * from exact
""")
res = collect(res)
MU = [x[1] for x in res]
OBS = [x[2] for x in res]

@info mean(MU), std(MU)
@info mean(OBS), std(OBS)
close(db)