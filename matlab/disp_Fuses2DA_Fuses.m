fprintf('Fuses2DA: best relaxed cost: %g\n',fuses2DAIterationsData(end, 2))
fprintf('   Fuses: best relaxed cost: %g\n\n',min(fusesIterationsData(:, 2)))

fprintf('   Fuses: best rounded cost: %g\n',min(fusesIterationsData(:, 4)))
fprintf('Fuses2DA: best rounded cost: %g\n',min(fuses2DAIterationsData(:, 4)))
fprintf('   Optimal cost: %g\n', fval_exact(i))

