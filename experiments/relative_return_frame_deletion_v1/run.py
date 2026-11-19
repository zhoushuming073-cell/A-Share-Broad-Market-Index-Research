import deletion
import monte_carlo
import comparisons
import inference
import verification
import report
if __name__=="__main__":
    deletion.build()
    monte_carlo.run()
    comparisons.build()
    inference.run()
    print(verification.verify(),flush=True)
    report.build()
